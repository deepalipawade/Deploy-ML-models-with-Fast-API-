from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Visit
# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/predict
# http://127.0.0.1:8000/redoc

# Initialize FastAPI app
app = FastAPI()

# Define the data model for input using Pydantic
class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Define the prediction endpoint
@app.post('/predict')
async def predict_species(iris: IrisSpecies):
    # Convert input data to dictionary
    data = iris.dict()
    
    # Load the trained model from disk
    loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
    
    # Prepare the input data for prediction
    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    
    # Make prediction
    prediction = loaded_model.predict(data_in)
    
    # Get the probability of the prediction
    probability = loaded_model.predict_proba(data_in).max()
    
    # Return the prediction and probability
    return {
        'prediction': prediction[0],
        'probability': probability
    }
