from fastapi import FastAPI, HTTPException
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseInformation(BaseModel):
  lat: float
  long: float
  balcony: int 
  loggia: int 
  veranda: int 
  kitchen: int 
  numberOfFloors: int
  floor: int 
  numberOfRooms: int 
  numberOfBedrooms: int 
  LotArea: int 
  ceilingHeight: int 
  numberOfBathrooms: int



class HousePricePredictor: 

    def __init__(self):
        self.ols: LinearRegression = load("./model_weights/ols.bin")
        self.ridge: Ridge = load("./model_weights/ridge.bin")
        self.lasso: Lasso = load("./model_weights/lasso.bin")
        self.bayesian: BayesianRidge = load("./model_weights/bayesian.bin")
        self.en: ElasticNet = load("./model_weights/en.bin")
        self.forest: ElasticNet = load("./model_weights/forest.bin")

    def predict(self, item: HouseInformation):
        return {
            "ols": self.ols.predict(self.__makePredictionModelFromObject(item))[0],
            "ridge": self.ridge.predict(self.__makePredictionModelFromObject(item))[0],
            "lasso": self.lasso.predict(self.__makePredictionModelFromObject(item))[0],
            "bayesian": self.bayesian.predict(self.__makePredictionModelFromObject(item))[0],
            "en": self.en.predict(self.__makePredictionModelFromObject(item))[0],
            "forest": self.en.predict(self.__makePredictionModelFromObject(item))[0],
        }

    def __makePredictionModelFromObject(self, house: HouseInformation): 
        predict_data = np.array([house.lat,house.long,house.balcony,house.loggia,house.veranda,house.kitchen,house.numberOfFloors,house.floor,house.numberOfRooms,house.numberOfBedrooms,house.LotArea,house.ceilingHeight,house.numberOfBathrooms])
        return predict_data.reshape(1, -1)



predictor = HousePricePredictor()

@app.get("")
def root():
    return {"GoTo": "/docs"}


@app.post("/api/v1/predict")
def is_user_item(request: HouseInformation):
    try:
        return {"response": predictor.predict(request)}
    except:
        raise HTTPException(status_code=418, detail="Exceptions can't be handheld by a teapot")