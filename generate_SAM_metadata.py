import torch
from collections import defaultdict
import hoarder
import time
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv
import cv2

from segment_anything import sam_model_registry, SamPredictor

from hoarder.datasets.local_dataset import LocalDataset
from hoarder.utils.conservator_utils import create_box_annotation, create_polygon_annotation, create_point_annotation
from hoarder.datasets.datasets import get_dataset_by_name
from hoarder.utils.conservator_utils import get_label_sets, get_label_id_from_name

print("Imported all libraries")

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./pretrained_checkpoint/sam_hq_vit_h.pth"

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


# Pull dataset
rd = get_dataset_by_name("SAM-trainval-1", fields=["name"])
ld = LocalDataset.clone(rd, clone_path='train/data/')
ld.pull()
ld.download()
ld.checkout('master')

# Get label sets
label_sets = get_label_sets()
valid_labels = []
for label in label_sets:
    if label.name == "LifeguardSegment": # NOTE: Change this based on preferred label sets
        valid_labels = [l.name for l in label.labels]

print(valid_labels)
label_id_dict = {l: get_label_id_from_name(l, label_set_name="LifeguardSegment", label_sets=label_sets) for l in valid_labels}
print(label_id_dict)

if False:
    print("Commiting first changes")
    ld.add_local_changes()
    ld.commit(f'Attempting to add annotations to images')
    ld.pull()

user_id = hoarder.conservator.get_user().id
image_frames = ld.get_frames()
img_annotations = defaultdict(list)
num_uploads = 1
for img_idx, frame in tqdm(enumerate(image_frames), total=len(image_frames)):
    annotations = frame['annotations']
    to_remove = []
    for i, ann in enumerate(annotations):
        if "cal-" in ann["labels"][0] or "calib_" in ann["labels"][0] or "water_intersect" in ann["labels"][0] or "sailor" in ann["labels"][0] or "dog" in ann["labels"][0]:
            to_remove.append(i)
    for i in to_remove[::-1]:
        del annotations[i]

for img_idx, frame in tqdm(enumerate(image_frames), total=len(image_frames)):
    image_path = ld.get_local_image_path(frame)
    annotations = frame['annotations']

    to_remove = []
    for i, ann in enumerate(annotations):
        if "cal-" in ann["labels"][0] or "calib_" in ann["labels"][0] or "water_intersect" in ann["labels"][0] or "sailor" in ann["labels"][0] or "dog" in ann["labels"][0]:
            to_remove.append(i)
    for i in to_remove[::-1]:
        del annotations[i]

    already_annotated = False
    for ann in annotations:
        if "boundingPolygon" in ann and ann["labels"][0] not in ["pool", "roi", "jacuzzi"]:
            already_annotated = True
            print(f"Already annotated {ann['labels'][0]}")
            break
    if already_annotated:
        continue
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_shape = image_np.shape
    sam_predictor.set_image(image_np)

    generated_metadata = {
        "boxes": {}, # xywh
        "polygons": {}, # xy,xy
    }

    for idx, ann in enumerate(annotations):
        if "boundingBox" in ann:
            # ld.pull()
            box = ann["boundingBox"]
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["w"])
            h = int(box["h"])

            label = ann["labels"][0]
            # NOTE: Skip invalid labels (pool, intercepts, etc.)
            if label not in valid_labels:
                continue

            input_box = torch.tensor([[x, y, x+w, y+h]], device=device)
            transformed_box = sam_predictor.transform.apply_boxes_torch(input_box, image_shape[:2])
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False,
                hq_token_only=False,
            )
            masks = masks.squeeze().detach().cpu().numpy() # (H, W)
            polygons = sv.mask_to_polygons(masks) # [np.ndarray([N, 2])]
            if len(polygons) > 0:
                polygons = polygons[0].tolist() # [(x,y), (x,y)]
            else:
                continue
            dict_polygons = [{"x": x_, "y": y_} for x_, y_ in polygons]
            polygon_annotation = create_polygon_annotation(dict_polygons, label, label_id_dict[label], user_id, as_dict=True)
            img_annotations[img_idx].append(polygon_annotation)
    if img_idx % 5000 == 0 and len(img_annotations):
        print(f"Adding {len(img_annotations)} annotations to dataset")
        ld.add_annotations_to_frames(img_annotations)
        ld.add_local_changes()
        ld.commit(f'Added new polygon annotations using sam-hq-h {num_uploads}')
        num_uploads += 1
        ld.push_commits()
        img_annotations = defaultdict(list)
if len(img_annotations):
    ld.add_annotations_to_frames(img_annotations)

ld.add_local_changes()
ld.commit('Added new polygon annotations using sam-hq-h')
ld.push_commits()
#ld.pull()