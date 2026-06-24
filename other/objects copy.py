import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

# Realsense setup
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# YOLO setup
model = YOLO("best_3.pt")
class_names = model.names  # Store model classes names

try:
    # Start streaming
    pipeline.start(config)

    

    output = []

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if  not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape

        for result in model.predict(source=color_image, stream=True, show=False, save=False):
            boxes = result.boxes.xywh  # bbox in formato (x_center, y_center, w, h)
            classes = result.boxes.cls  # indice della classe
            ids = result.boxes.id       # tracking ID (None se non disponibile)

            frame_output = []

            for i, box in enumerate(boxes):
                cx, cy, w, h = box

                cls_id = int(classes[i])
                class_name = class_names[cls_id]

                frame_output.append({
                    'class': class_name,
                    'bbox_center': (cx.item(), cy.item())
                })

            output.append(frame_output)
            #print(frame_output)

            # Show annotated image continuously
            annotated_frame = result.plot()
            cv2.imshow('RealSense YOLO', annotated_frame)

        # Interrupt the cycle if ANY key is pressed on the keyboard
        if cv2.waitKey(1) != -1:
            break
except Exception as e:
    print(e)
finally:
    # Stop streaming
    pipeline.stop()

    
