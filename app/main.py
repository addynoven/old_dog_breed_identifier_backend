from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services import prediction

app = FastAPI()

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
        label_number = prediction.predict_dog_breed_from_url(request.image_url)
        return {"label_number": label_number}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="There was an error processing the image.")
