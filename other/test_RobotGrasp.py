import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d

# YOLO model setup
model = YOLO("best.pt")
class_names = model.names
TARGET_CLASS = "screwdriver" # Update this to your desired class

# 1. RealSense Pipeline Setup
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# --- NEW: RealSense Depth Filters for better depth quality ---
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_filling_filter = rs.hole_filling_filter()
# -------------------------------------------------------------

align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 2. Open3D Visualizer Setup
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="RealSense 3D Point Cloud", width=1280, height=720)
main_pcd = o3d.geometry.PointCloud()
geometry_added = False

cv2.namedWindow("RealSense RGB", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RealSense RGB", 960, 720)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame: continue

        # --- NEW: Apply Filters to the depth frame ---
        aligned_depth_frame = spatial_filter.process(aligned_depth_frame)
        aligned_depth_frame = temporal_filter.process(aligned_depth_frame)
        aligned_depth_frame = hole_filling_filter.process(aligned_depth_frame)
        # ---------------------------------------------

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()

        results = model(color_image, stream=True, verbose=False)
        
        object_found_this_frame = False

        for result in results:
            if result.masks is not None:
                for i, mask_data in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    if class_names[cls_id] == TARGET_CLASS:
                        object_found_this_frame = True
                        
                        mask = mask_data.cpu().numpy()
                        mask = cv2.resize(mask, (640, 480))

                        # Overlay mask on OpenCV image
                        colored_mask = np.zeros_like(display_image)
                        colored_mask[mask > 0.5] = [0, 255, 0]
                        cv2.addWeighted(colored_mask, 0.4, display_image, 1.0, 0, display_image)

                        mask_indices = np.where(mask > 0.5)
                        z = depth_image[mask_indices] * 0.001 
                        
                        valid = z > 0
                        z = z[valid]
                        u = mask_indices[1][valid]
                        v = mask_indices[0][valid]

                        x = (u - intr.ppx) * z / intr.fx
                        y = (v - intr.ppy) * z / intr.fy
                        
                        # --- FIX: Invert Y and Z axes to correct the "Opposite View" ---
                        points = np.stack((x, -y, -z), axis=-1)
                        # ---------------------------------------------------------------
                        
                        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        colors = color_image_rgb[v, u] / 255.0

                        temp_pcd = o3d.geometry.PointCloud()
                        temp_pcd.points = o3d.utility.Vector3dVector(points)
                        temp_pcd.colors = o3d.utility.Vector3dVector(colors)
                        temp_pcd, _ = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

                        # --- FIX: Safe geometry updating for variable point counts ---
                        if geometry_added:
                            vis.remove_geometry(main_pcd, reset_bounding_box=False)
                            
                        main_pcd.points = temp_pcd.points
                        main_pcd.colors = temp_pcd.colors
                        
                        vis.add_geometry(main_pcd, reset_bounding_box=(not geometry_added))
                        geometry_added = True
                        # -------------------------------------------------------------

        # --- FIX: Handle window closing cleanly ---
        if geometry_added:
            # poll_events returns False if the user clicked the 'X' on the Open3D window
            if not vis.poll_events(): 
                break 
            vis.update_renderer()

        cv2.imshow("RealSense RGB", display_image)
        
        # This works if the OpenCV window is focused
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()