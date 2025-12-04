import tkinter as tk
from tkinter.filedialog import askopenfilename
import kornia.io as Kio

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

