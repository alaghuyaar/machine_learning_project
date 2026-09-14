from fastapi import FastAPI,Depends,HTTPException
from pydantic import BaseModel,Field,ConfigDict
from src.data.cleaning import clean
from dotenv import load_dotenv
import joblib
from src.models.predict import load_artifacts,predict_new
import os
import pandas as pd

app = FastAPI()
load_dotenv()
artifact_path = os.getenv('ARTIFACT_PATH')
artifact = load_artifacts(artifact_path)

class InputModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age            :   float = Field(gt=0,lt=120)
    job            :   str = Field(min_length=1,max_length=50)
    marital        :   str = Field(min_length=1,max_length=50)
    education      :   str = Field(min_length=1,max_length=50)
    default        :   str = Field(min_length=1,max_length=50)
    housing        :   str = Field(min_length=1,max_length=50)
    loan           :   str = Field(min_length=1,max_length=50)
    contact        :   str = Field(min_length=1,max_length=50)
    month          :   str = Field(min_length=1,max_length=50)
    day_of_week    :   str = Field(min_length=1,max_length=50)
    duration       :   float = Field(ge=0)
    campaign       :   float
    pdays          :   float
    previous       :   float
    poutcome       :   str = Field(min_length=1,max_length=50)
    emp_var_rate   :   float = Field(alias='emp.var.rate')
    cons_price_idx :   float = Field(alias='cons.price.idx')
    cons_conf_idx  :   float = Field(alias='cons.conf.idx')
    euribor3m      :   float
    nr_employed    :   float = Field(alias='nr.employed')

class ResponseModel(BaseModel):
    prediction : int
    probability : float


@app.post('/predict',response_model=ResponseModel)
def get_input(data:InputModel):
    raw_df = pd.DataFrame([data.model_dump(by_alias=True)])
    try:
        cleaned_df = clean(raw_df)
        pred_df = predict_new(cleaned_df,artifact)
        return {'prediction':int(pred_df['prediction'].iloc[0]),'probability':int(pred_df['probability'].iloc[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))