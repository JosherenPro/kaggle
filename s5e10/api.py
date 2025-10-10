from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("./xgb.h5")

class Features(BaseModel):
    road_type:str = "urban"
    num_lanes:int = 2
    curvature:float = 0.06
    speed_limit:int = 35
    lighting:str = "daylight"
    weather:str = "rainy"
    road_signs_present:bool = False
    public_road:str = True
    time_of_day:str = "afternoon"
    holiday:bool = False
    school_season:bool = True
    num_reported_accidents:int = 1

class Label(BaseModel):
    accident_risk:float


@app.post("/", response_model=Label)
def predict(features: Features):
    label = model.predict(features)
    return label

