import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import database FIRST to avoid conflict with TensorFlow
from .database import get_supabase_client
# from .models import sql_models

from .services import prediction

# Check if running in Colab
IS_COLAB = os.environ.get("COLAB_BACKEND") == "1"

# Create database tables only if NOT in Colab
# if not IS_COLAB:
#     sql_models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    supabase = get_supabase_client()
    if supabase:
        print("\033[92m✔ Supabase Client Initialized\033[0m", flush=True)
    else:
        print("\033[93m⚠ Supabase Client Failed to Initialize\033[0m", flush=True)
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_url: str

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Dog Breed Identification API!"}

@app.post("/predict", tags=["Prediction"])
async def predict_breed(request: ImageRequest):
    try:
        print("Received request for prediction")
        label_number = prediction.predict_dog_breed_from_url(request.image_url)
        print("Prediction completed")
        return {"label_number": label_number}

    except Exception as e:
        print(f"Error during prediction: {e}")
        error_msg = str(e)
        if "No dog detected" in error_msg:
             raise HTTPException(status_code=400, detail=error_msg)
        raise HTTPException(status_code=500, detail="There was an error processing the image.")
