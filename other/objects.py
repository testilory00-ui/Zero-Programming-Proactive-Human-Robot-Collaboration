import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
from collections import defaultdict

# YOLO setup
model = YOLO("best_3.pt")
class_names = model.names  # nomi delle classi

# Parametri filtro
MIN_FRAMES = 7  # 👈 un oggetto deve essere visto almeno 5 frame per essere accettato

# Strutture dati
id_counter = defaultdict(int)     # conta quante volte vediamo ogni (classe, id)
confirmed_objects = set()         # oggetti confermati (classe, id)
output = []

source = r"C:\Users\testi\iCloudDrive\Polimi\Tesi\Immagini\parts_no_yolo.HEIC"
# Avvio tracking
for result in model.track(source=source, save=True, stream=True, show=True, persist=True):
    if result.boxes.id is None:
        continue

    boxes = result.boxes.xywh
    classes = result.boxes.cls
    ids = result.boxes.id

    frame_output = []

    for i, box in enumerate(boxes):
        cx, cy, w, h = box

        track_id = int(ids[i])
        cls_id = int(classes[i])
        class_name = class_names[cls_id]

        # aggiorna contatore per questo oggetto fisico
        id_counter[(class_name, track_id)] += 1

        # se supera la soglia, lo consideriamo valido
        if id_counter[(class_name, track_id)] >= MIN_FRAMES:
            confirmed_objects.add((class_name, track_id))

        frame_output.append({
            'class': class_name,
            'id': track_id,
            'bbox_center': (cx.item(), cy.item())
        })

    output.append(frame_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- Risultati finali ---
# Se vuoi solo la lista delle classi uniche (senza ID):
unique_classes = set([cls for cls, _ in confirmed_objects])
print("Classi trovate nel video:", list(unique_classes))

