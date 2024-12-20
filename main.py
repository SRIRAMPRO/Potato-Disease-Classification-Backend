from io import BytesIO

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.models import load_model

app=FastAPI()

origins=[
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL=load_model("model1.keras")
CLASS_NAMES=["Early Blight", "Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am alive"


def read_file_as_image(data) ->np.ndarray:
    return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict(
file: UploadFile = File(...)
):
    image=read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence':confidence
    }
    

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)