import kornia.augmentation as K

def process_image(image):

    normalize = K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    resize = K.Resize(size=(224, 224))

    image = resize(image)
    image = normalize(image)

    return image