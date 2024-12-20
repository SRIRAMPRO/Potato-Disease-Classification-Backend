from io import BytesIO

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image


app=FastAPI()

endpoint="http://localhost:8501/v1/models/my_model:predict"
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
    json_data={
        'instances':img_batch.tolist()
    }
    response=requests.post(endpoint,json=json_data)
    
    prediction=response.json()["predictions"][0]
    predicted_class=CLASS_NAMES[np.argmax(prediction)]
    confidence=round(np.max(prediction)*100,2)
    
    return{
        "Class":predicted_class,
        "Confidence":confidence
    }

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)