import torch 
import tkinter as tk
from tkinter.filedialog import askopenfilename
import kornia.io as Kio
from models.AlopeciaClassifier import AlopeciaClassifier
import matplotlib.pyplot as plt
import numpy as np


def load_image(device="cpu"):

    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = askopenfilename(
        title="Select an image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.avif")],
    )
    if not file_path:
        raise ValueError("No image selected.")
    
    image = Kio.load_image(file_path, device=device)
    return file_path, image

def show_image(file_path):
    image = plt.imread(file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlopeciaClassifier().to(device)
    model.eval()

    file_path, image = load_image(device=device)

    show_image(file_path)

    with torch.inference_mode():
        output = model(image)
    print(f"Androgenetic Alopecia severity: {output}")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()