import open3d as o3d
import numpy as np
import time
from typing import List, Dict, Optional, Tuple

class RobustUniversalGraspEstimator:
    def __init__(self, max_aperture: float = 0.04, jaw_depth: float = 0.03, 
                 jaw_thickness: float = 0.005, palm_thickness: float = 0.01, 
                 jaw_width_y: float = 0.02):
        self.max_aperture = max_aperture
        self.jaw_depth = jaw_depth
        self.jaw_thickness = jaw_thickness
        self.palm_thickness = palm_thickness
        self.jaw_width_y = jaw_width_y 
        
        self.jaw_pad_y = 0.016  
        self.jaw_pad_z = 0.015  
        
        print(f"[INIT] Gripper Config: Aperture={max_aperture*100:.1f}cm, Depth={jaw_depth*100:.1f}cm")

    def check_collisions(self, local_points: np.ndarray, grasp_width: float) -> bool:
        """Checks if the solid parts of the gripper intersect the object OR the table."""
        x, y, z = local_points[:, 0], local_points[:, 1], local_points[:, 2]
        half_w = grasp_width / 2.0
        
        # Palm Collision Check
        in_palm = (
            (z < -self.jaw_depth) & 
            (z > -self.jaw_depth - self.palm_thickness) &
            (np.abs(x) < half_w + self.jaw_thickness) &
            (np.abs(y) < self.jaw_width_y / 2.0)
        )
        if np.any(in_palm): return True

        # Jaws Collision Check (z < 0.005 prevents table penetration)
        in_jaws = (
            (z > -self.jaw_depth) & 
            (z < 0.005) & 
            (np.abs(y) < self.jaw_width_y / 2.0) & 
            (np.abs(x) > half_w + 0.002) & 
            (np.abs(x) < half_w + self.jaw_thickness + 0.002)
        )
        if np.any(in_jaws): return True

        return False

    def calculate_contact_points(self, local_points: np.ndarray, grasp_width: float) -> Tuple[int, int]:
        """Evaluates independent bilateral contact to allow for partial/curved grips."""
        half_w = grasp_width / 2.0
        # Increased tolerance from 3mm to 6mm to catch curved surfaces like tubes
        depth_tolerance = 0.006 

        left_pad_contact = (
            (local_points[:, 0] > half_w - depth_tolerance) & (local_points[:, 0] < half_w + depth_tolerance) &
            (np.abs(local_points[:, 1]) < self.jaw_width_y / 2.0) & 
            (local_points[:, 2] > -self.jaw_depth) & (local_points[:, 2] < 0.005)
        )
        
        right_pad_contact = (
            (local_points[:, 0] < -half_w + depth_tolerance) & (local_points[:, 0] > -half_w - depth_tolerance) &
            (np.abs(local_points[:, 1]) < self.jaw_width_y / 2.0) & 
            (local_points[:, 2] > -self.jaw_depth) & (local_points[:, 2] < 0.005)
        )
        return int(np.sum(left_pad_contact)), int(np.sum(right_pad_contact))

    def find_top_grasps(self, pcd: o3d.geometry.PointCloud, top_k: int = 5) -> List[Dict]:
        start_time = time.time()
        print("[PROCESS] Analyzing geometry with dynamic depth sliding...")
        
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array([0., 0., 1.0]))

        down_pcd = pcd.voxel_down_sample(voxel_size=0.003)
        pts = np.asarray(down_pcd.points)
        norms = np.asarray(down_pcd.normals)
        full_pts = np.asarray(pcd.points)
        
        pcd_tree = o3d.geometry.KDTreeFlann(down_pcd)
        valid_grasps = []
        
        object_pts = full_pts[full_pts[:, 2] > 0.002]
        com = np.mean(object_pts, axis=0) if len(object_pts) > 0 else np.array([0,0,0])

        num_seeds = min(len(pts), 3000)
        seed_indices = np.random.choice(len(pts), num_seeds, replace=False)
        gravity_vector = np.array([0.0, 0.0, -1.0])

        for i in seed_indices:
            p1, n1 = pts[i], norms[i]
            if p1[2] < 0.005: continue 
                
            [k, idx, _] = pcd_tree.search_radius_vector_3d(p1, self.max_aperture)
            
            for j in idx[1:]: 
                p2, n2 = pts[j], norms[j]
                if p2[2] < 0.005: continue 
                
                vec = p2 - p1
                dist = np.linalg.norm(vec)
                
                if dist < 0.002: continue 
                if np.dot(n1, n2) > -0.60 or abs(np.dot(vec / dist, n1)) < 0.60: 
                    continue

                center = (p1 + p2) / 2.0
                close_dir = vec / dist
                
                # Create an orthogonal basis to sample approach directions (allow the gripper to rotate!)
                ref = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(close_dir, ref)) > 0.95:
                    ref = np.array([1.0, 0.0, 0.0])
                u = np.cross(close_dir, ref)
                u /= np.linalg.norm(u)
                v = np.cross(u, close_dir)
                
                # Explore multiple rotations around the closing axis
                for angle in np.linspace(0, np.pi, 6, endpoint=False):
                    approach_dir = np.cos(angle) * u + np.sin(angle) * v
                    
                    # Ensure approach comes generally from the top hemisphere
                    if approach_dir[2] > 0:
                        approach_dir = -approach_dir

                    R_world_local = np.column_stack([close_dir, np.cross(approach_dir, close_dir), approach_dir])
                    collision_pts = full_pts[full_pts[:, 2] > 0.002]
                    
                    # Align width center
                    local_points_initial = (collision_pts - center) @ R_world_local  
                    in_grip_zone = (local_points_initial[:, 2] > -self.jaw_depth) & (local_points_initial[:, 2] < 0.01) & (np.abs(local_points_initial[:, 1]) < self.jaw_width_y / 2.0)
                    if not np.any(in_grip_zone): continue
                        
                    points_in_zone = local_points_initial[in_grip_zone]
                    min_x, max_x = np.min(points_in_zone[:, 0]), np.max(points_in_zone[:, 0])
                    true_width = max_x - min_x
                    if true_width > self.max_aperture: continue
                        
                    # Jaws cannot close tighter than the minimum aperture (5mm)
                    true_width = max(true_width, 0.005)
                    
                    center_shift_x = (max_x + min_x) / 2.0
                    snapped_center = center + close_dir * center_shift_x
                    
                    # ==============================================================
                    # NEW STRATEGY: DYNAMIC DEPTH SLIDING
                    # Plunge as deep as the target point, but pull back incrementally
                    # if collisions occur, finding the best partial-contact depth.
                    # ==============================================================
                    safe_center = None
                    for pull_back in np.arange(0.0, self.jaw_depth, 0.005):
                        test_center = snapped_center - approach_dir * pull_back
                        local_points = (collision_pts - test_center) @ R_world_local
                        
                        if not self.check_collisions(local_points, grasp_width=true_width):
                            safe_center = test_center
                            break # Found the deepest collision-free point!
                    
                    if safe_center is None: continue # Impossible to grasp without collision
                    
                    local_points = (collision_pts - safe_center) @ R_world_local
                    left_c, right_c = self.calculate_contact_points(local_points, grasp_width=true_width)
                    
                    # Require at least a small amount of contact on BOTH jaws (Bilateral constraint)
                    if left_c >= 2 and right_c >= 2:
                        dist_to_com = np.linalg.norm(safe_center - com)
                        alignment_penalty = 1.0 - np.dot(approach_dir, gravity_vector) 
                        total_contact = left_c + right_c
                        
                        # Score rewards deeper grasps (closer to COM) and penalizes bad alignments
                        score = (alignment_penalty * 500.0) + (dist_to_com * 100.0) - (total_contact * 0.02)
                        
                        valid_grasps.append({
                            "center": safe_center, "close": close_dir, "approach": approach_dir,
                            "width": true_width, "score": score, "contact": total_contact, 
                            "alignment": alignment_penalty, "R": R_world_local
                        })

        print(f"[RESULT] Found {len(valid_grasps)} valid grasps.")
        print(f"[PROCESS] Inference time: {time.time() - start_time:.3f} seconds")
        
        if not valid_grasps: return []
        valid_grasps.sort(key=lambda x: x["score"])
        return valid_grasps[:top_k]

    def create_gripper_visualization(self, grasp: Dict, color: List[float] = [0.3, 0.3, 0.3]) -> List[o3d.geometry.Geometry]:
        w, d, t, pt, wy = grasp["width"], self.jaw_depth, self.jaw_thickness, self.palm_thickness, self.jaw_width_y

        palm = o3d.geometry.TriangleMesh.create_box(width=w + 2*t, height=wy, depth=pt)
        palm.translate((-(w + 2*t)/2, -wy/2, -d - pt))

        left_jaw = o3d.geometry.TriangleMesh.create_box(width=t, height=wy, depth=d)
        left_jaw.translate((-w/2 - t, -wy/2, -d))

        right_jaw = o3d.geometry.TriangleMesh.create_box(width=t, height=wy, depth=d)
        right_jaw.translate((w/2, -wy/2, -d))

        pad_l = o3d.geometry.TriangleMesh.create_box(width=0.001, height=self.jaw_pad_y, depth=self.jaw_pad_z)
        pad_l.translate((-w/2 + 0.001, -self.jaw_pad_y/2, -d)).paint_uniform_color([1, 0, 0])
        
        pad_r = o3d.geometry.TriangleMesh.create_box(width=0.001, height=self.jaw_pad_y, depth=self.jaw_pad_z)
        pad_r.translate((w/2 - 0.002, -self.jaw_pad_y/2, -d)).paint_uniform_color([1, 0, 0])

        gripper = palm + left_jaw + right_jaw + pad_l + pad_r
        gripper.compute_vertex_normals()
        gripper.paint_uniform_color(color)

        T = np.eye(4)
        T[:3, :3] = grasp["R"]
        T[:3, 3] = grasp["center"]
        gripper.transform(T)

        app_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([grasp["center"] - grasp["approach"]*0.08, grasp["center"]]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        app_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]]) 

        return [gripper, app_line]


# ==========================================
# SIMULATION OBJECT GENERATORS 
# ==========================================
def add_table(mesh):
    table = o3d.geometry.TriangleMesh.create_box(width=0.40, height=0.40, depth=0.01)
    table.translate((-0.20, -0.20, -0.01)) 
    table.paint_uniform_color([0.2, 0.2, 0.2])
    return table + mesh

def generate_screwdriver():
    handle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.08)
    R = handle.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    handle.rotate(R, center=(0,0,0)).translate((0, 0, 0.015)) 
    handle.paint_uniform_color([0.9, 0.3, 0.0])
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=0.10)
    shaft.rotate(R, center=(0,0,0)).translate((0.09, 0, 0.015)) 
    shaft.paint_uniform_color([0.8, 0.8, 0.8])
    mesh = handle + shaft
    mesh.compute_vertex_normals()
    return add_table(mesh).sample_points_uniformly(25000, use_triangle_normal=True)

def generate_throttle_stop():
    base = o3d.geometry.TriangleMesh.create_box(width=0.04, height=0.03, depth=0.003)
    base.translate((-0.02, -0.015, 0.0)) 
    arm = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.04, depth=0.002)
    R = arm.get_rotation_matrix_from_xyz((np.pi/4, 0, 0))
    arm.rotate(R, center=(0,0,0)).translate((0.01, 0.0, 0.003))
    mesh = base + arm
    mesh.paint_uniform_color([0.7, 0.7, 0.5])
    mesh.compute_vertex_normals()
    return add_table(mesh).sample_points_uniformly(25000, use_triangle_normal=True)

def generate_hollow_tube():
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.08)
    R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    mesh.rotate(R, center=(0,0,0)).translate((0, 0, 0.015))
    pcd = mesh.sample_points_uniformly(25000, use_triangle_normal=True)
    pts, norms = np.asarray(pcd.points), np.asarray(pcd.normals)
    side_mask = np.abs(norms[:, 0]) < 0.95 
    tube_pcd = o3d.geometry.PointCloud()
    tube_pcd.points = o3d.utility.Vector3dVector(pts[side_mask])
    tube_pcd.normals = o3d.utility.Vector3dVector(norms[side_mask])
    tube_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    table = o3d.geometry.TriangleMesh.create_box(width=0.4, height=0.4, depth=0.01).translate((-0.2, -0.2, -0.01)).paint_uniform_color([0.2, 0.2, 0.2])
    return tube_pcd + table.sample_points_uniformly(10000)

def generate_u_channel():
    base = o3d.geometry.TriangleMesh.create_box(width=0.08, height=0.04, depth=0.005)
    left_wall = o3d.geometry.TriangleMesh.create_box(width=0.08, height=0.01, depth=0.02)
    right_wall = o3d.geometry.TriangleMesh.create_box(width=0.08, height=0.01, depth=0.02)
    left_wall.translate((0, 0, 0.005))
    right_wall.translate((0, 0.035, 0.005))
    channel = base + left_wall + right_wall
    R = channel.get_rotation_matrix_from_xyz((0, 0, np.pi/6)) 
    channel.rotate(R, center=(0,0,0)).translate((-0.04, -0.02, 0))
    channel.paint_uniform_color([0.4, 0.5, 0.6])
    channel.compute_vertex_normals()
    return add_table(channel).sample_points_uniformly(25000, use_triangle_normal=True)

def generate_tilted_bracket():
    leg1 = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.04, depth=0.04)
    leg2 = o3d.geometry.TriangleMesh.create_box(width=0.04, height=0.01, depth=0.04)
    leg2.translate((0.01, 0, 0))
    bracket = leg1 + leg2
    R = bracket.get_rotation_matrix_from_xyz((np.pi/8, -np.pi/12, np.pi/4))
    bracket.rotate(R, center=(0,0,0))
    pts = np.asarray(bracket.vertices)
    min_z = np.min(pts[:, 2])
    bracket.translate((0, 0, -min_z))
    bracket.paint_uniform_color([0.8, 0.4, 0.2])
    bracket.compute_vertex_normals()
    return add_table(bracket).sample_points_uniformly(25000, use_triangle_normal=True)

def generate_upright_hollow_cylinder():
    outer_r, inner_r, height = 0.030, 0.026, 0.080
    outer = o3d.geometry.TriangleMesh.create_cylinder(radius=outer_r, height=height, resolution=60)
    inner = o3d.geometry.TriangleMesh.create_cylinder(radius=inner_r, height=height + 0.002, resolution=60)
    
    outer.translate((0, 0, height/2))
    inner.translate((0, 0, height/2))
    
    outer_pcd = outer.sample_points_uniformly(20000)
    inner_pcd = inner.sample_points_uniformly(20000)
    
    out_pts, out_n = np.asarray(outer_pcd.points), np.asarray(outer_pcd.normals)
    in_pts, in_n = np.asarray(inner_pcd.points), np.asarray(inner_pcd.normals)
    
    # Remove top and bottom caps by filtering out vertical normals
    out_mask = np.abs(out_n[:, 2]) < 0.9
    in_mask = np.abs(in_n[:, 2]) < 0.9
    
    # Add a flat rim at the top to ensure a robust graspable surface
    theta = np.linspace(0, 2*np.pi, 200)
    r_ring = np.linspace(inner_r, outer_r, 5)
    T, R = np.meshgrid(theta, r_ring)
    rim_x = (R * np.cos(T)).flatten()
    rim_y = (R * np.sin(T)).flatten()
    rim_z = np.full_like(rim_x, height)
    rim_pts = np.column_stack([rim_x, rim_y, rim_z])
    rim_n = np.zeros_like(rim_pts)
    rim_n[:, 2] = 1.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((out_pts[out_mask], in_pts[in_mask], rim_pts)))
    pcd.normals = o3d.utility.Vector3dVector(np.vstack((out_n[out_mask], -in_n[in_mask], rim_n)))
    pcd.paint_uniform_color([0.2, 0.6, 0.8])
    
    return add_table(pcd)

if __name__ == "__main__":
    model = RobustUniversalGraspEstimator(max_aperture=0.03)
    
    SELECTED_DEMO = "throttle_stop" 
    
    DEMO_OBJECTS = {
        "screwdriver": generate_screwdriver,
        "throttle_stop": generate_throttle_stop,
        "hollow_tube": generate_hollow_tube,
        "u_channel": generate_u_channel,
        "tilted_bracket": generate_tilted_bracket,
        "upright_hollow_cylinder": generate_upright_hollow_cylinder,
    }

    pcd = DEMO_OBJECTS[SELECTED_DEMO]()
    top_grasps = model.find_top_grasps(pcd, top_k=1)
    
    if top_grasps:
        print(f"\n[SUCCESS] {len(top_grasps)} Grasps Found!")
        geometries = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]
        
        colors = [
            [0.1, 0.8, 0.1],  # 1st - Green
            [0.8, 0.8, 0.1],  # 2nd - Yellow
            [0.8, 0.5, 0.1],  # 3rd - Orange
            [0.8, 0.1, 0.1],  # 4th - Red
            [0.5, 0.1, 0.5],  # 5th - Purple
        ]
        
        for idx, grasp in enumerate(top_grasps):
            print(f" > Rank {idx+1}: Score: {grasp['score']:.2f}, Width: {grasp['width']*100:.2f}cm, Contacts: {grasp['contact']}")
            c = colors[idx] if idx < len(colors) else [0.3, 0.3, 0.3]
            geometries.extend(model.create_gripper_visualization(grasp, color=c))
            
        o3d.visualization.draw_geometries(geometries)
    else:
        print("\n[FAILED] Could not find a valid grasp.")
        o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)])