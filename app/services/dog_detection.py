from ultralytics import YOLO
from PIL import Image
import os

# Load the YOLO12 Small model
# It will download 'yolo12s.pt' automatically on first use if not present
model = YOLO("yolo12s.pt")

def detect_dog(image: Image.Image) -> bool:
    """
    Detects if there is a dog in the image using YOLO12.
    Returns True if a dog is detected, False otherwise.
    """
    # Run inference
    results = model(image, verbose=False)
    
    # Check results for 'dog' class
    # COCO dataset class index for 'dog' is 16
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'dog':
                print("Dog detected!")
                return True
                
    return False
