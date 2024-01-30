import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
with open('models/pipline_model.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    Result: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.get('/topfeatures')
def version():
    return model['top_features']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'Result': y[0]
    }


if __name__ == '__main__':
    uvicorn.run(app)
