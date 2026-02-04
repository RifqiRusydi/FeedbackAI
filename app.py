from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="Feedback Classification API")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    X = vectorizer.transform([data.text])
    prediction = model.predict(X)[0]
    return {"category": prediction}

##http://127.0.0.1:8000/docs#/default/predict_predict_post