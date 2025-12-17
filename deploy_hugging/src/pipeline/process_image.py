import kornia.augmentation as K
from kornia.utils import image_to_tensor

def process_image(image, device="cpu"):
    # Kornia expects images in [0-1] range and float values
    # Without .copy(), torch throws a warning
    image = image_to_tensor(image.copy()).float() / 255.0
    image = image.to(device)  

    # Apply normalization and resizing
    normalize = K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    resize = K.Resize(size=(224, 224))

    image = resize(image)
    image = normalize(image)
    return image
