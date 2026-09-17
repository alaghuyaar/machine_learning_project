from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field,ConfigDict
from src.data.cleaning import clean
from dotenv import load_dotenv
from src.models.predict import load_artifacts,predict_new
import os
import pandas as pd
from utils.log import config_log  #for running the config file
import logging
from src.exceptions import DataCleaningError,PredictionError

app = FastAPI()
load_dotenv()
config_log()
artifact_path = os.getenv('ARTIFACT_PATH')
artifact = load_artifacts(artifact_path)
logger = logging.getLogger(__name__)

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
        return {'prediction':pred_df['prediction'].iloc[0],'probability':(pred_df['probability'].iloc[0]).item()}

    except DataCleaningError as e:
        raise HTTPException(status_code=400, detail='Bad Request Error')
    
    except PredictionError as e:
        raise HTTPException(status_code=500, detail='Internal server error')
    
    except Exception as e:
        logger.error('Unhandled error during prediction: %s', str(e))
        raise HTTPException(status_code=500, detail='Internal server error')