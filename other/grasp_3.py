"""
Top-down parallel-jaw grasp estimator for tabletop objects — VLM Integration Rewrite.
Generates top-K spatially distinct grasps, exports to JSON, and renders a VLM-ready image.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GripperGeometry:
    max_aperture:   float = 0.040   
    jaw_depth:      float = 0.030   
    jaw_thickness:  float = 0.005   
    palm_thickness: float = 0.010   
    jaw_half_width: float = 0.010   

    @property
    def jaw_full_width(self) -> float:
        return 2.0 * self.jaw_half_width


@dataclass
class GraspCandidate:
    center:    np.ndarray           
    close_dir: np.ndarray           
    approach:  np.ndarray           
    width:     float                
    contact:   int                  
    score:     float = field(default=0.0)

    @property
    def lateral(self) -> np.ndarray:
        return np.cross(self.approach, self.close_dir)

    @property
    def rotation(self) -> np.ndarray:
        return np.column_stack([self.close_dir, self.lateral, self.approach])


# ─────────────────────────────────────────────────────────────────────────────
# Core planner
# ─────────────────────────────────────────────────────────────────────────────

class GraspPlanner:
    def __init__(
        self,
        gripper: GripperGeometry | None = None,
        *,
        voxel_size:      float = 0.003,
        min_z:           float = 0.005,
        num_seeds:       int   = 500,
        n_approaches:    int   = 6,
        min_contact:     int   = 4,
        min_line:        float = 0.008,
        squeeze_tol:     float = 0.0015,
        normal_dot_max:  float = -0.35,
        min_descent:     float = 0.50,
        score_weights:   dict | None = None,
    ) -> None:
        self.gripper         = gripper or GripperGeometry()
        self.voxel_size      = voxel_size
        self.min_z           = min_z
        self.num_seeds       = num_seeds
        self.n_approaches    = n_approaches
        self.min_contact     = min_contact
        self.min_line        = min_line
        self.squeeze_tol     = squeeze_tol
        self.normal_dot_max  = normal_dot_max
        self.min_descent     = min_descent
        self.weights = {"descent": 2.0, "com": 0.1, "contact": -0.01}
        if score_weights:
            self.weights.update(score_weights)

    def plan(self, pcd: o3d.geometry.PointCloud, top_k: int = 5, nms_dist: float = 0.025) -> list[GraspCandidate]:
        """Return the top-K spatially distinct grasps."""
        bar = "─" * 62
        print(f"\n{bar}")
        t_wall = time.perf_counter()

        t0  = time.perf_counter()
        pcd = self._ensure_normals(pcd)
        seed_cloud = pcd.voxel_down_sample(self.voxel_size)

        seed_pts   = np.asarray(seed_cloud.points,  dtype=np.float32)
        seed_norms = np.asarray(seed_cloud.normals, dtype=np.float32)

        # Fast evaluation: use a downsampled cloud for matrix ops instead of the raw dense pcd
        eval_pcd   = pcd.voxel_down_sample(max(0.002, self.voxel_size * 0.5))
        obj_pts    = np.asarray(eval_pcd.points,    dtype=np.float32)
        obj_pts    = obj_pts[obj_pts[:, 2] > self.min_z]

        if len(obj_pts) == 0:
            return []

        com  = obj_pts.mean(axis=0)
        tree = o3d.geometry.KDTreeFlann(seed_cloud)

        above   = seed_pts[:, 2] > self.min_z
        horiz   = np.sqrt(np.maximum(0.0, 1.0 - seed_norms[:, 2] ** 2)) > 0.30
        valid   = np.where(above & horiz)[0]
        rng     = np.random.default_rng(0)
        seeds   = rng.choice(valid, min(self.num_seeds, len(valid)), replace=False)

        candidates: list[GraspCandidate] = []
        g = self.gripper

        for i in seeds:
            p1, n1 = seed_pts[i], seed_norms[i]
            _, nbr_raw, _ = tree.search_radius_vector_3d(p1, g.max_aperture)
            nbr_idx = np.asarray(nbr_raw[1:], dtype=np.int32)
            if len(nbr_idx) == 0: continue

            P2, N2 = seed_pts[nbr_idx], seed_norms[nbr_idx]
            alive = P2[:, 2] > self.min_z
            P2, N2 = P2[alive], N2[alive]

            vecs  = P2 - p1
            dists = np.linalg.norm(vecs, axis=1)
            ok    = dists > 1e-3
            vecs, dists, P2, N2 = vecs[ok], dists[ok], P2[ok], N2[ok]
            if len(vecs) == 0: continue

            close_dirs = vecs / dists[:, None]
            ok  = (N2 @ n1) < self.normal_dot_max
            ok &= np.abs(close_dirs @ n1) >= 0.40
            if not np.any(ok): continue

            close_dirs_f, P2_f = _dedup_dirs(close_dirs[ok], P2[ok])

            for cd, p2 in zip(close_dirs_f, P2_f):
                approaches = self._sample_approaches(cd)
                for app in approaches:
                    g_result = self._evaluate_grasp(p1, p2, cd, app, obj_pts, com)
                    if g_result is not None:
                        candidates.append(g_result)

        if not candidates:
            return []

        # Sort all by score (lower is better)
        candidates.sort(key=lambda c: c.score)

        # NMS (Non-Maximum Suppression) based on distance
        filtered_grasps = []
        for c in candidates:
            if not filtered_grasps:
                filtered_grasps.append(c)
                continue
            
            # Ensure the new grasp is at least `nms_dist` meters away from all accepted ones
            dists = [np.linalg.norm(c.center - f.center) for f in filtered_grasps]
            if min(dists) > nms_dist:
                filtered_grasps.append(c)
                
            if len(filtered_grasps) == top_k:
                break

        print(f"  Total pipeline time: {(time.perf_counter()-t_wall)*1000:6.1f} ms")
        print(f"  Returned Top {len(filtered_grasps)} distinct grasps.")
        print(bar)
        return filtered_grasps

    def _sample_approaches(self, close_dir: np.ndarray) -> list[np.ndarray]:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(close_dir, ref))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u = np.cross(close_dir, ref).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(u, close_dir).astype(np.float32)

        results = []
        angles = np.linspace(0.0, 2 * np.pi, self.n_approaches, endpoint=False)
        for a in angles:
            app = (float(np.cos(a)) * u + float(np.sin(a)) * v).astype(np.float32)
            app /= np.linalg.norm(app)
            if float(app[2]) > 0: app = -app
            if float(app[2]) <= -self.min_descent: results.append(app)

        down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        proj = down - float(np.dot(down, close_dir)) * close_dir
        mag  = float(np.linalg.norm(proj))
        if mag > 1e-4:
            best_down = (proj / mag).astype(np.float32)
            if float(best_down[2]) <= -self.min_descent: results.append(best_down)
        return results

    def _evaluate_grasp(self, p1, p2, close_dir, approach, obj_pts, com) -> GraspCandidate | None:
        g = self.gripper
        center  = (p1 + p2) * 0.5
        lateral = np.cross(approach, close_dir).astype(np.float32)
        lateral /= np.linalg.norm(lateral)

        R = np.vstack([close_dir, lateral, approach])
        local = ((obj_pts - center) @ R.T).astype(np.float32)

        lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

        in_open = ((lz > -g.jaw_depth) & (lz < 0.002) & (np.abs(ly) < g.jaw_half_width))
        if not np.any(in_open): return None

        x_open = lx[in_open]
        x_min, x_max = float(x_open.min()), float(x_open.max())
        true_width = x_max - x_min
        if true_width > g.max_aperture: return None

        shift  = (x_max + x_min) * 0.5
        center = center + close_dir * shift
        lx     = lx - shift
        hw     = true_width * 0.5

        if self._collides(lx, ly, lz, hw, g): return None
        contact, ok = self._check_contact(lx, ly, lz, hw, g)
        if contact < self.min_contact or not ok: return None

        w = self.weights
        score = (w["descent"] * (1.0 + float(approach[2])) 
                 + w["com"] * float(np.linalg.norm(center - com)) 
                 + w["contact"] * contact)
                 
        return GraspCandidate(center=center, close_dir=close_dir, approach=approach, width=true_width, contact=contact, score=score)

    def _check_contact(self, lx, ly, lz, hw, g) -> tuple[int, bool]:
        stol, ml = self.squeeze_tol, self.min_line
        in_y = np.abs(ly) < g.jaw_half_width
        in_z = (lz > -g.jaw_depth) & (lz < 0.002)

        def _jaw_stats(face_x: float) -> tuple[int, bool]:
            mask = in_y & in_z & (np.abs(lx - face_x) < stol)
            n = int(np.sum(mask))
            if n < 2: return n, False
            ok = (float(ly[mask].max() - ly[mask].min()) >= ml) or (float(lz[mask].max() - lz[mask].min()) >= ml)
            return n, ok

        n_right, ok_right = _jaw_stats(+hw)
        n_left,  ok_left  = _jaw_stats(-hw)
        return n_right + n_left, ok_right and ok_left

    @staticmethod
    def _collides(lx, ly, lz, hw, g) -> bool:
        in_y = np.abs(ly) < g.jaw_half_width
        in_jaw_z = (lz > -g.jaw_depth) & (lz < 0.0)
        abs_x = np.abs(lx)
        if np.any(in_y & in_jaw_z & (abs_x > hw) & (abs_x < hw + g.jaw_thickness)): return True
        in_palm_z = (lz > -g.jaw_depth - g.palm_thickness) & (lz < -g.jaw_depth)
        if np.any(in_y & in_palm_z & (abs_x < hw + g.jaw_thickness)): return True
        return False

    @staticmethod
    def _ensure_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 1.0]))
        return pcd

def _dedup_dirs(close_dirs, P2, bin_size=0.10):
    seen, keep_cd, keep_p2, inv = set(), [], [], 1.0 / bin_size
    for cd, p2 in zip(close_dirs, P2):
        key = (int(round(cd[0] * inv)), int(round(cd[1] * inv)))
        if key not in seen:
            seen.add(key)
            keep_cd.append(cd); keep_p2.append(p2)
    if not keep_cd: return np.empty((0, 3)), np.empty((0, 3))
    return np.array(keep_cd, dtype=np.float32), np.array(keep_p2, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# VLM Helpers (Export & Thick Fork Visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def export_to_json(grasps: list[GraspCandidate], filepath: str = "top_grasps.json"):
    data = []
    for i, g in enumerate(grasps):
        data.append({
            "id": i,
            "score": float(g.score),
            "width": float(g.width),
            "center": g.center.tolist(),
            "close_dir": g.close_dir.tolist(),
            "approach": g.approach.tolist()
        })
    with open(filepath, 'w') as f:
        json.dump({"grasps": data}, f, indent=4)
    print(f"  Saved JSON data to '{filepath}'")

def create_thick_fork(grasp: GraspCandidate, g: GripperGeometry, color: list[float], radius=0.002) -> list[o3d.geometry.TriangleMesh]:
    """Draws the gripper as thick cylindrical lines (a fork) so VLMs can easily see them."""
    c, hw, d = grasp.center, grasp.width * 0.5, g.jaw_depth
    R = grasp.rotation
    parts = []
    
    def add_cylinder(p_start, p_end):
        vec = p_end - p_start
        height = np.linalg.norm(vec)
        if height < 1e-4: return
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        cyl.paint_uniform_color(color)
        
        # Align Z-axis with vector
        vec_norm = vec / height
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, vec_norm)
        angle = np.arccos(np.clip(np.dot(z_axis, vec_norm), -1.0, 1.0))
        
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cyl.rotate(rot_mat, center=(0,0,0))
        elif np.dot(z_axis, vec_norm) < 0:
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0,0,0))
            
        cyl.translate((p_start + p_end) / 2.0)
        parts.append(cyl)

    # Calculate corner points in world frame
    left_tip  = c + R @ np.array([-hw, 0, 0])
    left_root = c + R @ np.array([-hw, 0, -d])
    right_tip = c + R @ np.array([hw, 0, 0])
    right_root= c + R @ np.array([hw, 0, -d])
    palm_mid  = c + R @ np.array([0, 0, -d])
    handle    = c + R @ np.array([0, 0, -d - 0.05])
    
    # Draw the fork lines
    add_cylinder(left_root, left_tip)    # Left Jaw
    add_cylinder(right_root, right_tip)  # Right Jaw
    add_cylinder(left_root, right_root)  # Base bridge
    add_cylinder(palm_mid, handle)       # Handle/Approach line
    
    return parts

# ─────────────────────────────────────────────────────────────────────────────
# Scene creation (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _make_table():
    t = o3d.geometry.TriangleMesh.create_box(0.40, 0.40, 0.01)
    t.translate((-0.20, -0.20, -0.01))
    t.paint_uniform_color([0.18, 0.18, 0.18])
    return t

def _sample(mesh, n):
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(n, use_triangle_normal=True)

def _with_table(obj_pcd: o3d.geometry.PointCloud, table_n: int = 8_000) -> o3d.geometry.PointCloud:
    return obj_pcd + _sample(_make_table(), table_n)

def scene_screwdriver():
    R90y = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0])
    handle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.080)
    handle.rotate(R90y, center=(0, 0, 0)); handle.translate((0.0, 0.0, 0.015))
    handle.paint_uniform_color([0.85, 0.30, 0.05])
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=0.100)
    shaft.rotate(R90y, center=(0, 0, 0)); shaft.translate((0.09, 0.0, 0.015))
    shaft.paint_uniform_color([0.80, 0.80, 0.80])
    return _sample(handle + shaft, 22_000) + _sample(_make_table(), 8_000)

def scene_curved_torus() -> o3d.geometry.PointCloud:
    torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.035, tube_radius=0.012)
    R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 4, np.pi / 3, np.pi / 6])
    torus.rotate(R, center=(0, 0, 0))
    min_z = np.min(np.asarray(torus.vertices)[:, 2])
    torus.translate((0.0, 0.0, -min_z + 0.002))
    torus.paint_uniform_color([0.80, 0.30, 0.60])
    return _with_table(_sample(torus, 20_000))

def scene_hex_bolt() -> o3d.geometry.PointCloud:
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.008, height=0.040, resolution=32)
    shaft.translate((0, 0, 0.020)); shaft.paint_uniform_color([0.72, 0.72, 0.72])
    head = o3d.geometry.TriangleMesh.create_cylinder(radius=0.014, height=0.013, resolution=6)
    head.translate((0, 0, 0.0465)); head.paint_uniform_color([0.55, 0.55, 0.55])
    return _with_table(_sample(shaft + head, 20_000))


def scene_l_bracket() -> o3d.geometry.PointCloud:
    fa = o3d.geometry.TriangleMesh.create_box(0.060, 0.010, 0.025)
    fa.translate((-0.030, -0.005, 0.0)); fa.paint_uniform_color([0.60, 0.62, 0.65])
    fb = o3d.geometry.TriangleMesh.create_box(0.010, 0.050, 0.025)
    fb.translate((0.020, -0.005, 0.0)); fb.paint_uniform_color([0.60, 0.62, 0.65])
    return _with_table(_sample(fa + fb, 20_000))


def scene_u_channel() -> o3d.geometry.PointCloud:
    wall_h, wall_t, base_w, length = 0.020, 0.003, 0.060, 0.060
    base = o3d.geometry.TriangleMesh.create_box(length, base_w, wall_t)
    base.translate((-length / 2, -base_w / 2, 0.0))
    lw = o3d.geometry.TriangleMesh.create_box(length, wall_t, wall_h)
    lw.translate((-length / 2, -base_w / 2, wall_t))
    rw = o3d.geometry.TriangleMesh.create_box(length, wall_t, wall_h)
    rw.translate((-length / 2, base_w / 2 - wall_t, wall_t))
    mesh = base + lw + rw; mesh.paint_uniform_color([0.78, 0.80, 0.85])
    return _with_table(_sample(mesh, 24_000))


def scene_hollow_tube() -> o3d.geometry.PointCloud:
    outer_r, inner_r, height = 0.025, 0.020, 0.080
    outer = o3d.geometry.TriangleMesh.create_cylinder(radius=outer_r, height=height, resolution=60)
    outer.translate((0, 0, height / 2))
    inner = o3d.geometry.TriangleMesh.create_cylinder(radius=inner_r, height=height + 0.002, resolution=60)
    inner.translate((0, 0, height / 2))
    
    outer_pcd = _sample(outer, 15_000)
    inner_pcd = _sample(inner, 15_000)
    
    out_pts, out_n = np.asarray(outer_pcd.points), np.asarray(outer_pcd.normals)
    in_pts, in_n = np.asarray(inner_pcd.points), np.asarray(inner_pcd.normals)
    
    # Remove top and bottom caps by filtering out vertical normals
    out_mask = np.abs(out_n[:, 2]) < 0.9
    in_mask = np.abs(in_n[:, 2]) < 0.9
    
    # Add a flat rim at the top to ensure a robust graspable surface
    theta = np.linspace(0, 2 * np.pi, 200)
    r_ring = np.linspace(inner_r, outer_r, 5)
    T, R = np.meshgrid(theta, r_ring)
    rim_x, rim_y = (R * np.cos(T)).flatten(), (R * np.sin(T)).flatten()
    rim_pts = np.column_stack([rim_x, rim_y, np.full_like(rim_x, height)])
    rim_n = np.column_stack([np.zeros_like(rim_x), np.zeros_like(rim_x), np.ones_like(rim_x)])
    
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np.vstack((out_pts[out_mask], in_pts[in_mask], rim_pts)))
    obj_pcd.normals = o3d.utility.Vector3dVector(np.vstack((out_n[out_mask], -in_n[in_mask], rim_n)))
    obj_pcd.paint_uniform_color([0.52, 0.58, 0.63])
    return _with_table(obj_pcd)

def scene_dumbbell_bracket() -> o3d.geometry.PointCloud:
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.08)
    shaft.translate((0, 0, 0.04))
    end1 = o3d.geometry.TriangleMesh.create_box(0.06, 0.06, 0.02)
    end1.translate((-0.03, -0.03, 0.0))
    end2 = o3d.geometry.TriangleMesh.create_box(0.06, 0.06, 0.02)
    end2.translate((-0.03, -0.03, 0.08))
    knob = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    knob.translate((0.035, 0.0, 0.05))
    mesh = shaft + end1 + end2 + knob
    mesh.paint_uniform_color([0.30, 0.70, 0.50])
    R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, np.pi / 6, 0])
    mesh.rotate(R, center=(0, 0, 0))
    min_z = np.min(np.asarray(mesh.vertices)[:, 2])
    mesh.translate((0.0, 0.0, -min_z + 0.002))
    return _with_table(_sample(mesh, 25_000))

def scene_c_clamp() -> o3d.geometry.PointCloud:
    spine = o3d.geometry.TriangleMesh.create_box(0.010, 0.050, 0.060)
    spine.translate((-0.005, -0.025, 0.0))
    top_jaw = o3d.geometry.TriangleMesh.create_box(0.040, 0.050, 0.008)
    top_jaw.translate((-0.005, -0.025, 0.052))
    bot_jaw = o3d.geometry.TriangleMesh.create_box(0.040, 0.050, 0.008)
    bot_jaw.translate((-0.005, -0.025, 0.0))
    mesh = spine + top_jaw + bot_jaw; mesh.paint_uniform_color([0.30, 0.30, 0.32])
    return _with_table(_sample(mesh, 24_000))

def scene_cone_plug() -> o3d.geometry.PointCloud:
    base = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.020)
    base.translate((0, 0, 0.010))
    top = o3d.geometry.TriangleMesh.create_cone(radius=0.015, height=0.030)
    top.translate((0, 0, 0.020))
    mesh = base + top
    mesh.paint_uniform_color([0.80, 0.30, 0.30])
    return _with_table(_sample(mesh, 20_000))

def scene_cross() -> o3d.geometry.PointCloud:
    b1 = o3d.geometry.TriangleMesh.create_box(0.080, 0.015, 0.015)
    b1.translate((-0.040, -0.0075, 0.0))
    b2 = o3d.geometry.TriangleMesh.create_box(0.015, 0.080, 0.015)
    b2.translate((-0.0075, -0.040, 0.0))
    mesh = b1 + b2
    mesh.paint_uniform_color([0.20, 0.70, 0.30])
    return _with_table(_sample(mesh, 20_000))

def scene_mug() -> o3d.geometry.PointCloud:
    outer_r, inner_r, height = 0.020, 0.016, 0.040
    outer = o3d.geometry.TriangleMesh.create_cylinder(radius=outer_r, height=height, resolution=48)
    outer.translate((0, 0, height / 2))
    inner = o3d.geometry.TriangleMesh.create_cylinder(radius=inner_r, height=height + 0.002, resolution=48)
    inner.translate((0, 0, height / 2))
    outer_pcd = _sample(outer, 15_000)
    inner_pcd = _sample(inner, 10_000)
    inner_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(inner_pcd.normals))
    
    # Increased handle dimensions for easier grasping
    handle_v = o3d.geometry.TriangleMesh.create_box(0.008, 0.012, 0.025)
    handle_v.translate((0.035, -0.006, 0.005))
    handle_h1 = o3d.geometry.TriangleMesh.create_box(0.018, 0.012, 0.008)
    handle_h1.translate((0.018, -0.006, 0.022))
    handle_h2 = o3d.geometry.TriangleMesh.create_box(0.018, 0.012, 0.008)
    handle_h2.translate((0.018, -0.006, 0.005))
    handle_pcd = _sample(handle_v + handle_h1 + handle_h2, 8_000)
    
    obj_pcd = outer_pcd + inner_pcd + handle_pcd
    obj_pcd.paint_uniform_color([0.30, 0.50, 0.80])
    return _with_table(obj_pcd)

def scene_cup_with_curved_handle() -> o3d.geometry.PointCloud:
    """Creates a cup with a C-shaped handle made of overlapping spheres."""
    # 1. Create cup body (hollow cylinder)
    outer_r, inner_r, height, base_h = 0.04, 0.035, 0.08, 0.005
    
    outer = o3d.geometry.TriangleMesh.create_cylinder(radius=outer_r, height=height, resolution=60)
    outer.translate((0, 0, height / 2))
    
    inner_h = height - base_h
    inner = o3d.geometry.TriangleMesh.create_cylinder(radius=inner_r, height=inner_h, resolution=60)
    inner.translate((0, 0, base_h + inner_h / 2))
    
    outer_pcd = _sample(outer, 15000)
    inner_pcd = _sample(inner, 12000)
    
    out_pts, out_n = np.asarray(outer_pcd.points), np.asarray(outer_pcd.normals)
    in_pts, in_n = np.asarray(inner_pcd.points), np.asarray(inner_pcd.normals)
    
    # Remove top caps
    out_top_mask = (out_pts[:, 2] > height - 0.001) & (out_n[:, 2] > 0.9)
    in_top_mask = (in_pts[:, 2] > height - 0.001) & (in_n[:, 2] > 0.9)
    
    out_mask = ~out_top_mask
    in_mask = ~in_top_mask
    
    # Add a flat rim at the top
    theta = np.linspace(0, 2 * np.pi, 200)
    r_ring = np.linspace(inner_r, outer_r, 5)
    T, R = np.meshgrid(theta, r_ring)
    rim_x, rim_y = (R * np.cos(T)).flatten(), (R * np.sin(T)).flatten()
    rim_pts = np.column_stack([rim_x, rim_y, np.full_like(rim_x, height)])
    rim_n = np.column_stack([np.zeros_like(rim_x), np.zeros_like(rim_x), np.ones_like(rim_x)])
    
    cup_pts = np.vstack((out_pts[out_mask], in_pts[in_mask], rim_pts))
    # Flip normals for the inner surface
    cup_n = np.vstack((out_n[out_mask], -in_n[in_mask], rim_n))
    
    cup_pcd = o3d.geometry.PointCloud()
    cup_pcd.points = o3d.utility.Vector3dVector(cup_pts)
    cup_pcd.normals = o3d.utility.Vector3dVector(cup_n)
    
    # 2. Create a curved handle from a series of spheres
    handle_mesh = o3d.geometry.TriangleMesh()
    handle_arc_radius = 0.025
    handle_offset_x = outer_r - 0.005 # Start handle slightly inside the cup wall
    handle_height_center = height * 0.5
    
    for angle in np.linspace(-np.pi/2.2, np.pi/2.2, 12):
        x = handle_offset_x + handle_arc_radius * np.cos(angle)
        z = handle_height_center + handle_arc_radius * np.sin(angle)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sphere.translate((x, 0, z))
        handle_mesh += sphere
        
    handle_mesh.merge_close_vertices(0.001)
    handle_pcd = _sample(handle_mesh, 5000)
    
    # 3. Combine and color
    obj_pcd = cup_pcd + handle_pcd
    obj_pcd.paint_uniform_color([0.3, 0.5, 0.8])
    
    return _with_table(obj_pcd)

# ─────────────────────────────────────────────────────────────────────────────
# Scene registry
# ─────────────────────────────────────────────────────────────────────────────

SCENES: dict[str, tuple] = {
    "screwdriver":   (scene_screwdriver,   "Cylindrical handle lying on side"),
    "curved_torus":  (scene_curved_torus,  "Torus rotated at a strange angle"),
    "hex_bolt":      (scene_hex_bolt,      "Upright bolt with hex head"),
    "l_bracket":     (scene_l_bracket,     "Right-angle metal bracket flat"),
    "u_channel":     (scene_u_channel,     "U-shaped aluminium channel, opening up"),
    "hollow_tube":   (scene_hollow_tube,   "Thick-walled hollow tube (vertical)"),
    "dumbbell_bracket": (scene_dumbbell_bracket, "Large complex geometry with boxes and a sphere"),
    "c_clamp":       (scene_c_clamp,       "C-clamp body – tests concave geometry"),
    "cone_plug":     (scene_cone_plug,     "Cone on top of a cylinder base"),
    "cross":         (scene_cross,         "Two intersecting boxes forming a +"),
    "mug":           (scene_mug,           "Hollow cylinder with a side handle"),
    "cup_with_curved_handle": (scene_cup_with_curved_handle, "Cup with a curved C-shaped handle"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SELECTED = "cup_with_curved_handle"   # ← change to test a different scene

    fn, description = SCENES[SELECTED]
    print(f"\n{'═'*62}")
    print(f"  Object : {SELECTED}  –  {description}")
    print(f"{'═'*62}")

    gripper = GripperGeometry(
        max_aperture   = 0.040,
        jaw_depth      = 0.030,
        jaw_thickness  = 0.005,
        palm_thickness = 0.010,
        jaw_half_width = 0.010,
    )
    planner = GraspPlanner(
        gripper        = gripper,
        voxel_size     = 0.004,    # Increased from 0.003 to limit seeds
        min_z          = 0.005,
        num_seeds      = 300,      # Reduced from 500
        n_approaches   = 4,        # Reduced from 6
        min_contact    = 4,
        min_line       = 0.008,
        squeeze_tol    = 0.0015,
        normal_dot_max = -0.35,
        min_descent    = 0.50,
        score_weights  = {"descent": 2.0, "com": 0.1, "contact": -0.01},
    )

    # 1. Generate Point Cloud
    pcd = fn()

    # 2. Plan and get Top K grasps
    top_grasps = planner.plan(pcd, top_k=5, nms_dist=0.03)

    if not top_grasps:
        print("Failed to find grasps.")
        exit()

    # 3. Export to JSON for the VLM/Robot pipeline
    export_to_json(top_grasps, "vlm_grasps.json")

    # 4. Prepare geometries for VLM Image (Color-coded forks)
    # Bright distinct colors for the VLM to distinguish
    VLM_COLORS = [
        [1.0, 0.0, 0.0],  # 0: Red
        [0.0, 1.0, 0.0],  # 1: Green
        [0.0, 0.5, 1.0],  # 2: Blue
        [1.0, 1.0, 0.0],  # 3: Yellow
        [1.0, 0.0, 1.0],  # 4: Magenta
    ]

    geometries = [pcd]
    for i, grasp in enumerate(top_grasps):
        color = VLM_COLORS[i % len(VLM_COLORS)]
        geometries.extend(create_thick_fork(grasp, gripper, color))

    # 5. Render, auto-capture screenshot, and show
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VLM Grasp Selection", width=1280, height=720)
    for geom in geometries:
        vis.add_geometry(geom)

    # Initialize view
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    lookat = pcd.get_center()

    # Define three distinct views for the VLM
    camera_views = [
        {"name": "top", "front": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0], "zoom": 0.65},
        {"name": "iso_right", "front": [1.0, 1.0, 0.8], "up": [0.0, 0.0, 1.0], "zoom": 0.75},
        {"name": "iso_left", "front": [-1.0, 1.0, 0.8], "up": [0.0, 0.0, 1.0], "zoom": 0.75},
    ]

    captured_images = []
    for view in camera_views:
        ctr.set_lookat(lookat)
        ctr.set_front(view["front"])
        ctr.set_up(view["up"])
        ctr.set_zoom(view["zoom"])
        
        # Poll events multiple times to ensure camera updates properly before capture
        for _ in range(5):
            vis.poll_events()
            vis.update_renderer()
            
        img_name = f"vlm_grasps_{view['name']}.png"
        vis.capture_screen_image(img_name)
        captured_images.append(img_name)
        print(f"  Saved screenshot to '{img_name}'.")
        
    print(f"  (You can now feed these {len(captured_images)} images and 'vlm_grasps.json' to your VLM).")
    print("  Close the Open3D window to exit.")
    
    vis.run()
    vis.destroy_window()