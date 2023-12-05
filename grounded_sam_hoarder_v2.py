import os
import json
import cv2
import numpy as np
from tqdm.notebook import tqdm
import supervision as sv
import torch
from PIL import Image


import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import (
    mask_to_rle_pytorch
)

from hoarder.datasets.local_dataset import LocalDataset
from hoarder.utils.conservator_utils import get_label_sets, get_label_id_from_name


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_image, Model

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./pretrained_checkpoint/sam_hq_vit_h.pth"
FINETUNED_MASKED_CHECKPOINT_PATH = "./train/work_dirs/hq_sam_h/epoch_4.pth"
GDINO_CKPT_PATH = "pretrained_checkpoint/groundingdino_swint_ogc.pth"
GDINO_CONFIG_PATH = "pretrained_checkpoint/GroundingDINO_SwinT_OGC.py"

finetune = False

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

if finetune:
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.mask_decoder.load_state_dict(torch.load(FINETUNED_MASKED_CHECKPOINT_PATH))
else:
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)

sam.to(device=device)
sam_predictor = SamPredictor(sam)

grounding_dino_model = Model(
    model_config_path=GDINO_CONFIG_PATH,
    model_checkpoint_path=GDINO_CKPT_PATH,
    device = "cuda",
)

ld = LocalDataset(path='sample_data/TestSAM')
ld_frames = ld.get_frames()

label_sets = get_label_sets()
label_name2ids = {}
for label in label_sets:
    if label.name == "LifeguardSegment":
        label_name2ids = {l.name:l.id for l in label.labels}

label_ids2idx = {id:idx for idx, id in enumerate(list(label_name2ids.values()))}

sample_index = 10

frame_info = ld_frames[sample_index]
image_path = ld.get_local_image_path(frame_info)

image_np = cv2.imread(image_path)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
image_shape = image_np.shape
sam_predictor.set_image(image_np)

annotations = frame_info['annotations']
generated_metadata = {
    "boxes": [],
    "masks": [],
    "class_id": [],
}
for idx, ann in enumerate(annotations):
    if "boundingBox" in ann:
        box = ann["boundingBox"]
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["w"])
        h = int(box["h"])

        label = ann["labels"][0]
        if label not in list(label_name2ids.keys()):
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
        masks = masks.squeeze().detach().cpu().numpy() # [H,W]
        generated_metadata["boxes"].append([x, y, x+w, y+h])
        generated_metadata["masks"].append(masks)
        generated_metadata["class_id"].append(label_ids2idx[label_name2ids[label]])

detections = sv.Detections(
    xyxy=np.array(generated_metadata["boxes"]),
    mask=np.array(generated_metadata["masks"]),
    class_id=np.array(generated_metadata["class_id"])
)

#assert sv.__version__ == '0.16.0'
#polygon_annotator = sv.PolygonAnnotator()
#annotated_frame = polygon_annotator.annotate(scene=image_np.copy(), detections=detections)
#sv.plot_image(annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
sv.plot_image(annotated_frame)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image_np.copy(), detections=detections)
sv.plot_image(annotated_frame)

TEXT_PROMPT = "person"
BOX_THRESHOLD = 0.245
TEXT_THRESHOLD = 0.287

# load image
sam_predictor.set_image(image_np)

# detect objects
detections, phrases = grounding_dino_model.predict_with_caption(
    image=image_np,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=BOX_THRESHOLD
)
class_id = Model.phrases2classes(phrases=phrases, classes=[TEXT_PROMPT])
detections.class_id = class_id

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=[TEXT_PROMPT])

sv.plot_image(annotated_image)

boxes_xyxy = torch.Tensor(detections.xyxy)
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(device)
masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )

mask_annotator = sv.MaskAnnotator()

if isinstance(masks, torch.Tensor):
    masks = masks.cpu().numpy().squeeze(1)
detections.mask = masks

annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
sv.plot_image(annotated_image)
print("done.")