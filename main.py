from src.pipeline.load_image import load_image
from src.pipeline.crop_head import crop_head
from src.pipeline.plot_head import plot_head
from src.pipeline.process_image import process_image
from src.pipeline.load_model import load_model
from src.pipeline.infer import infer

import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_path, image = load_image()
    cropped_image = crop_head(image)
    plot_head(cropped_image)
    processed_image = process_image(cropped_image, device=device)
    model = load_model(device=device)
    score = infer(model, processed_image)

    print(f"Androgenetic Alopecia severity: {score}")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()