from fastapi import FastAPI, Response
from pydantic import BaseModel

from services.crop_desease_service import predict_crop_disease
from services.crop_type_service import predict_crop, get_crop_type

class CropPredictionRequest(BaseModel):
    image: str

class CropDiseasePredictionRequest(BaseModel):
    image: str
    cropType: str

app = FastAPI()


@app.get("/healthz")
async def root():
    return {"message": "agriguard service is running!"}

@app.post("/cropType/predict")
async def predict_crop_type(requestBody: CropPredictionRequest):
    crop_index = predict_crop(requestBody.image)
    crop_type, confidence = get_crop_type(crop_index)
    return {"predictedCrop": crop_type, "confidence": confidence}

@app.post("/cropDisease/predict")
async def post_crop_disease(requestBody: CropDiseasePredictionRequest):
    crop_type = requestBody.cropType
    try:
        result = predict_crop_disease(requestBody.image, crop_type)
        return result
    except ValueError as e:
        return Response({}, status_code=400)
    