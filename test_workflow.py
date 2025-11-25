import sys
import os

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.prediction import predict_dog_breed_from_url
from app.database import SessionLocal
from app.models.sql_models import ImageLog

def test_workflow():
    # 1. Test Dog Image
    dog_url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_100.jpg"
    print(f"\n--- Testing Dog Image: {dog_url} ---")
    try:
        label = predict_dog_breed_from_url(dog_url)
        print(f"Result: {label}")
    except Exception as e:
        print(f"Error: {e}")

    # Verify DB
    db = SessionLocal()
    log = db.query(ImageLog).filter(ImageLog.image_url == dog_url).first()
    if log:
        print(f"DB Log: ID={log.id}, IsDog={log.is_dog}, Label={log.breed_label}")
    else:
        print("DB Log not found!")
    db.close()

    # 2. Test Cache Hit (Run again)
    print(f"\n--- Testing Cache Hit: {dog_url} ---")
    try:
        label = predict_dog_breed_from_url(dog_url)
        print(f"Result: {label}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Test Non-Dog Image
    non_dog_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    print(f"\n--- Testing Non-Dog Image: {non_dog_url} ---")
    try:
        label = predict_dog_breed_from_url(non_dog_url)
        print(f"Result: {label}")
    except Exception as e:
        print(f"Expected Error: {e}")

    # Verify DB
    db = SessionLocal()
    log = db.query(ImageLog).filter(ImageLog.image_url == non_dog_url).first()
    if log:
        print(f"DB Log: ID={log.id}, IsDog={log.is_dog}, Label={log.breed_label}")
    else:
        print("DB Log not found!")
    db.close()

if __name__ == "__main__":
    # Ensure DB tables exist
    from app.database import engine
    from app.models import sql_models
    sql_models.Base.metadata.create_all(bind=engine)
    
    test_workflow()
