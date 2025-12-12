import cv2
from mtcnn import MTCNN

def crop_head(image, margin=0.5):

    # Resize image if it's too large for faster processing
    if max(image.shape) > 1024:
        H, W = image.shape[:2]
        ratio = W / H 
        image = cv2.resize(image, (1024, int(1024 / ratio)))

    # Initialize the FaceDetector and detect faces
    detector = MTCNN()
    face_boxes = detector.detect_faces(image=image,
                                        threshold_onet=0.7)
          
        
    # Check if we have any boxes
    if len(face_boxes) == 0:
        return image  # Return the original image if no faces are detected

    # Get the first detected face box
    x1, y1, width, height = face_boxes[0]["box"] # x, y, width, height
    x2 = x1 + width
    y2 = y1 + height

    H, W = image.shape[:2]

    # Ensure coordinates are within image bounds    
    x1 = max(0, int(x1 - margin * (x2 - x1)))  # Move x1 left to include more of the head
    y1 = max(0, int(y1 - margin * (y2 - y1)))  # Move y1 down to include more of the head
    x2 = min(W, int(x2 + margin * (x2 - x1)))  # Move x2 right to include more of the head
    y2 = min(H, int(y2 + margin * (y2 - y1)))  # Move y2 down to include more of the head

    # Adjust to make the crop square for resizing purposes
    side_length = max(x2 - x1, y2 - y1)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Calculate new square coordinates
    half_side = side_length // 2
    
    x1 = max(0, center_x - half_side)
    y1 = max(0, center_y - half_side)
    x2 = min(W, center_x + half_side)
    y2 = min(H, center_y + half_side)

    # Crop the image using the adjusted coordinates           
    cropped_face = image[y1:y2, x1:x2, :]

    return cropped_face