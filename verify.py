import torch
import torchvision
import cv2
import pycocotools
import segment_geospatial
import rasterio
import matplotlib
import PIL
import skimage
import detectron2

print("Torch version:", torch._version_)
print("Torchvision version:", torchvision._version_)
print("OpenCV version:", cv2._version_)
print("COCO tools version:", pycocotools._version_)
print("Segment Geospatial version:", segment_geospatial._version_)
print("Rasterio version:", rasterio._version_)
print("Matplotlib version:", matplotlib._version_)
print("Pillow version:", PIL._version_)
print("Scikit-Image version:", skimage._version_)
print("Detectron2 version:", detectron2._version_)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")