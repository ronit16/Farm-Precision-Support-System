import os
import json
import random
import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator


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

def register_dataset():
    DatasetCatalog.register(f"farm_test", lambda: get_farm_dicts("./farm-data/test", "./farm-data/test/_annotations.coco.json"))
    MetadataCatalog.get(f"farm_test").set(thing_classes=["farm", "Baren-Land", "farm"])  # Add all category names here
    print(f"Successfully registered farm_test dataset")

def cfg_setup():   
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = (f"farm_test",)
    cfg.MODEL.WEIGHTS = "./model/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 1 class (0-1

    predictor = DefaultPredictor(cfg)
    return cfg, predictor

# Visualize predictions
def visualize_prediction(img_path, predictor, cfg):
    im = cv2.imread(img_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return im, out.get_image()[:, :, ::-1], outputs


def prediction_on_image_folder(input_dir, predictor, cfg):

    output_dir = os.getcwd() + "/output"
    os.makedirs(output_dir, exist_ok=True)
    output_mask_dir = os.getcwd() + "/output_masks"
    os.makedirs(output_mask_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            original_img, predicted_img, outputs = visualize_prediction(img_path, predictor, cfg)
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, predicted_img)
            print(f"Processed: {img_name}")

            # Get the masks
            if outputs["instances"].has("pred_masks"):
                masks = outputs["instances"].pred_masks.to("cpu").numpy()

                # Create binary mask
                if masks.size > 0:
                    binary_mask = np.zeros_like(masks[0], dtype=np.uint8)
                    for mask in masks:
                        binary_mask = np.logical_or(binary_mask, mask)
                    binary_mask = (binary_mask * 255).astype(np.uint8)
                else:
                    binary_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
            else:
                binary_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)

            # Save the mask
            mask_path = os.path.join(output_mask_dir, img_name)
            cv2.imwrite(mask_path, binary_mask)
            print(f"Saved mask: {img_name}")

            area_calculation(outputs, binary_mask, img_name)

    print("Processing complete. Output images are saved in the output_images directory.")


def prediction_on_tiff(input_image, img_name, predictor, cfg,):

    output_dir = os.getcwd() + "/output"
    os.makedirs(output_dir, exist_ok=True)
    output_mask_dir = os.getcwd() + "/output_masks"
    os.makedirs(output_mask_dir, exist_ok=True)
   
    tiff_image=Image.open(input_image)
    image = np.array(tiff_image)

    # Ensure the image is in BGR format as OpenCV expects
    if image.shape[2] == 4:  # If the image has an alpha channel
        image = image[:, :, :3]
    image = image[:, :, ::-1]  # Convert RGB to BGR

    original_img, predicted_img, outputs = visualize_prediction(input_image, predictor, cfg)

    output_path = os.path.join(output_dir, img_name)
    if output_path.endswith(".tif"):
        output_path=output_path.replace(".tif",".jpg")
    else:
        output_path=output_path.replace(".tiff",".jpg")
    cv2.imwrite(output_path, predicted_img)
    print(f"Processed: ",output_path)

    # Get the masks
    if outputs["instances"].has("pred_masks"):
        masks = outputs["instances"].pred_masks.to("cpu").numpy()

        # Create binary mask
        if masks.size > 0:
            binary_mask = np.zeros_like(masks[0], dtype=np.uint8)
            for mask in masks:
                binary_mask = np.logical_or(binary_mask, mask)
            binary_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
    else:
        binary_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)

    # Save the mask
    mask_path = os.path.join(output_mask_dir, img_name)
    if mask_path.endswith(".tif"):
        mask_path=mask_path.replace(".tif",".jpg")
    else:
        mask_path=mask_path.replace(".tiff",".jpg")
    cv2.imwrite(mask_path, binary_mask)
    print(f"Saved mask:",mask_path)

    print("Processing complete. Output images are saved in the output and output_masks directory.")

    return outputs, binary_mask

def area_calculation(outputs, binary_mask, img_name):
    # Calculate the area of each instance
    resolution = 0.5  # meters per pixel
    instance_details = []
    pred_boxes = outputs["instances"].pred_boxes.tensor.numpy()
    scores = outputs["instances"].scores.numpy()
    pred_classes = outputs["instances"].pred_classes.numpy()
    pred_masks = outputs["instances"].pred_masks.numpy()

    for i in range(len(pred_boxes)):
        area = np.sum(pred_masks[i]) * (resolution ** 2)
        instance = {
            "bbox": pred_boxes[i].tolist(),
            "score": float(scores[i]),
            "category_id": int(pred_classes[i]),
            "area": area
        }
        instance_details.append(instance)

    output_instance_details_path = os.getcwd() + "/output_masks/" + img_name + ".json"
    with open(output_instance_details_path, 'w') as f:
        json.dump(instance_details, f, indent=4)

    print(f"Saved instance details: {output_instance_details_path}")

    # Total farm area in the image 
    total_area = np.sum(binary_mask) * (resolution ** 2)
    print(f"Total farm area in the image: {total_area} sq. meters")

# Function to get georeferencing info from a TIFF file
def get_georeferencing_info(tif_path):
    with rasterio.open(tif_path) as dataset:
        transform = dataset.transform
        crs = dataset.crs
    return transform, crs

# Function to georeference a mask image
def geo_reference_mask(mask, output_tif_path, transform, crs):
    # Get the dimensions of the mask
    height, width = mask.shape[:2]

    # Create a new rasterio dataset
    new_transform = from_origin(transform.c, transform.f, transform.a, -transform.e)
    with rasterio.open(
        output_tif_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        crs=crs,
        transform=new_transform,
    ) as dst:
        dst.write(mask, 1)

def main():
    cfg, predictor = cfg_setup()
    choice=int(input("Press 1 if you want to test on jpg images in farm-data/test folder \n Press 2 if you want to test on tiff images in data folder"))
    if choice==1:
        # Get the path to the input image directory
        input_dir = "./farm-data/test"
        prediction_on_image_folder(input_dir, predictor, cfg)
    elif choice==2:
        # Prediction on Tiff image
        tiff_folder_path = "./data"
        for img_name in os.listdir(tiff_folder_path):
            if img_name.endswith(('.tif', '.tiff')):
                outputs, binary_mask = prediction_on_tiff(os.path.join(tiff_folder_path, img_name),img_name, predictor, cfg)
                area_calculation(outputs, binary_mask, img_name)
                transform, crs = get_georeferencing_info(os.path.join(tiff_folder_path,img_name))
                os.makedirs("./output_georeferenced",exist_ok=True)
                output_georeferenced_mask_path = os.path.join("./output_georeferenced", img_name)
                geo_reference_mask(binary_mask, output_georeferenced_mask_path, transform, crs)

    else:
        print("Invalid choice! Please choose either 1 or 2.")
                

if __name__ == "__main__":
    main()