import os
import json
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import cv2
import numpy as np

def get_farm_dicts(img_dir, annot_file):
    with open(annot_file, 'r') as f:
        dataset = json.load(f)

    dataset_dicts = []
    category_ids = set()
    for img in dataset['images']:
        record = {}

        filename = os.path.join(img_dir, img['file_name'])
        if not os.path.exists(filename):
            print(f"Warning: Image file does not exist: {filename}")
            continue

        try:
            image = cv2.imread(filename)
            if image is None:
                print(f"Warning: Unable to read image {filename}. File may be corrupted.")
                continue
            height, width = image.shape[:2]
        except Exception as e:
            print(f"Error reading image {filename}: {str(e)}")
            continue

        record["file_name"] = filename
        record["image_id"] = img['id']
        record["height"] = height
        record["width"] = width

        annos = [anno for anno in dataset['annotations'] if anno['image_id'] == img['id']]
        objs = []
        for anno in annos:
            category_ids.add(anno['category_id'])
            obj = {
                "bbox": anno['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": anno['segmentation'],
                "category_id": anno['category_id'],
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    print(f"Unique category IDs found: {sorted(category_ids)}")
    return dataset_dicts

for d in ["train", "test"]:
    DatasetCatalog.register("farm_" + d, lambda d=d: get_farm_dicts("./farm-data" + d, "./farm-data" + d + "/_annotations.coco.json"))
    MetadataCatalog.get("farm_" + d).set(thing_classes=["farm", "Baren-Land", "farm"])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("farm_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000  # Adjust according to your dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Only one class (farm)
cfg.OUTPUT_DIR = "./model/"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Training completed. Model saved to", cfg.OUTPUT_DIR)
