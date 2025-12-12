import easygui
import cv2
from matplotlib.image import imread
import numpy as np

def load_image():
    file_path = easygui.fileopenbox(
        title="Select an image",
        filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp", "*.avif"]
    )
    if not file_path:
        raise ValueError("No image selected.")
    
    image = imread(file_path)
    return file_path, image

