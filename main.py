from src.pipeline.crop_head import crop_head
from src.pipeline.process_image import process_image
from src.pipeline.load_model import load_model
from src.pipeline.infer import infer

import gradio as gr
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_alopecia(image, device=device):
    cropped_image = crop_head(image)
    processed_image = process_image(cropped_image, device=device)
    model = load_model(device=device)
    score = infer(model, processed_image)
    return score

interface = gr.Interface(
    fn=predict_alopecia,
    inputs=gr.Image(
        type="numpy",     
        label="Upload image"
    ),
    outputs=gr.Slider(label="Analysis Result", minimum=0, maximum=6, step=1),
    title="Androgenetic Alopecia Classification",
    description="Upload or take a selfie or a photo of the scalp to analyze the androgenetic alopecia severity from 0 to 6",
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(share=True)

