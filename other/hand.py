import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
from collections import deque

# ========== Config ==========
MODEL_PATH = "hand_landmarker.task"
NUM_HANDS = 2

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result, state):
  if detection_result is None or not detection_result.hand_landmarks:
     return rgb_image
  
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    hand_label = handedness[0].category_name

    # Stato della mano corrente ("open" / "pinch")
    state_text = hand_states.get(hand_label, "")
    
    cv2.putText(annotated_image, state_text,
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def extract_relevant_keypoints(detection_result):

    results = {}

    if detection_result is None or not detection_result.hand_landmarks or not detection_result.handedness:
        return results  # Nessuna mano rilevata
    
    for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
       # 'Left' o 'Right'
        hand_label = handedness[0].category_name

        # Landmark indices:
        wrist = hand_landmarks[0] # WRIST
        thumb = hand_landmarks[4] # THUMB_TIP
        index = hand_landmarks[8] # INDEX_FINGER_TIP

        results[hand_label] = {
            "wrist": (wrist.x, wrist.y, wrist.z),
            "thumb": (thumb.x, thumb.y, thumb.z),
            "index": (index.x, index.y, index.z)
        }
    
    return results

detection_result = None
def print_result(result, output_image, timestamp_ms):
    global detection_result
    detection_result = result

# create detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=NUM_HANDS,
    result_callback=print_result
)

landmarker = vision.HandLandmarker.create_from_options(options)

# capture webcam
cap = cv2.VideoCapture(0)

# Parameters for the sliding window
history = {"Left": deque(), "Right": deque()}
window = 600 # ms
threshold_dist = 0.08
last_process_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int(time.time() * 1000)
    
    landmarker.detect_async(mp_image, timestamp_ms)

    # extract results
    keypoints = extract_relevant_keypoints(detection_result)
    distance = {}
    wrist_action = {}
    for hand_label, points in keypoints.items():
        wx, wy, wz = points["wrist"]
        tx, ty, tz = points["thumb"]
        ix, iy, iz = points["index"]

        distance = math.sqrt((tx - ix)**2 + (ty - iy)**2 + (tz - iz)**2)
        wrist_position = math.sqrt(wx**2 + wy**2 + wz**2)
        
        history[hand_label].append((timestamp_ms, distance, wrist_position))

        while history[hand_label] and (timestamp_ms - history[hand_label][0][0]) > window:
            history[hand_label].popleft()

    if timestamp_ms - last_process_time >= window:
        hand_states = {}
        for hand_label, buffer in history.items():
            if buffer:
                dist = np.array([d for t, d, p in buffer], dtype=np.float32)
                pos = np.array([p for t, d, p in buffer], dtype=np.float32)
                t = np.array([t for t, d, p in buffer], dtype=np.float64)

                dt = (t[-1] - t[0])/1000

                mean_dist = np.mean(dist)
                grad_wrist = (pos[-1] - pos[0])/dt if dt>0 else 0.0
                
                if np.abs(grad_wrist) < 0.06: 
                    if mean_dist < threshold_dist:
                        hand_states[hand_label] = "pinch"
                    else:
                        hand_states[hand_label] = "nothing"
                else:
                    if mean_dist < threshold_dist:
                        hand_states[hand_label] = "move object"
                    else:
                        hand_states[hand_label] = "move without object"
            else:
                hand_states[hand_label] = ""

        last_process_time = timestamp_ms

    annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result, hand_states)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("MediaPipe Hands - Live", annotated_bgr)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()