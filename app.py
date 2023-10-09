import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))
prediction_labels = {0: 'Benign', 1: 'Malignant'}
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "prediction_text": ""})

@app.post('/predict')
async def predict(
    request: Request,
    perimeter_mean: float = Form(...),
    area_mean: float = Form(...),
    area_se: float = Form(...),
    perimeter_worst: float = Form(...),
    area_worst: float = Form(...)
):
    features = np.array([perimeter_mean, area_mean, area_se, perimeter_worst, area_worst]).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = prediction_labels[prediction[0]]

    return templates.TemplateResponse("prediction.html", {"request": request, "prediction_text": f"Diagnosis: {predicted_label}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
