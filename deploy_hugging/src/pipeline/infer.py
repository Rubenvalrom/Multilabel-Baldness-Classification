import torch

def infer(model, image):
    model.eval()
    with torch.inference_mode():
        output = model(image)

    output = output * 6
    output = torch.round(output).squeeze().long()
    output = output.squeeze().cpu().numpy()
    
    return output