from fastapi import FastAPI,HTTPException,Depends
from pydantic import BaseModel,Field,ConfigDict
from src.data.cleaning import clean
from dotenv import load_dotenv
from src.models.predict import load_artifacts,predict_new
import os
import pandas as pd
from utils.log import config_log  #for running the config file
import logging
from src.exceptions import DataCleaningError,PredictionError
from typing import Literal

app = FastAPI()
load_dotenv()
config_log()
artifact_path = os.getenv('ARTIFACT_PATH')
artifact = load_artifacts(artifact_path)
logger = logging.getLogger(__name__)

def get_artifact():
    return artifact

class InputModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age            :   float = Field(gt=0,lt=120)
    job            :   Literal['admin.', 'blue-collar', 'technician', 'services', 
                            'management', 'retired', 'entrepreneur', 'self-employed', 
                            'housemaid', 'unemployed', 'student', 'unknown'] = Field(min_length=1,max_length=50)
    marital        :   Literal['married', 'single', 'divorced', 'unknown'] = Field(min_length=1,max_length=50)
    education      :   Literal['university.degree', 'high.school', 'basic.9y', 
                            'professional.course', 'basic.4y', 'basic.6y', 
                            'unknown', 'illiterate'] = Field(min_length=1,max_length=50)
    default        :   Literal['no', 'unknown', 'yes'] = Field(min_length=1,max_length=50)
    housing        :   Literal['yes', 'no', 'unknown'] = Field(min_length=1,max_length=50)
    loan           :   Literal['no', 'yes', 'unknown'] = Field(min_length=1,max_length=50)
    contact        :   Literal['cellular', 'telephone'] = Field(min_length=1,max_length=50)
    month          :   Literal['may', 'jul', 'aug', 'jun', 
                            'nov', 'apr', 'oct', 'sep', 'mar', 'dec'] = Field(min_length=1,max_length=50)
    day_of_week    :   Literal['thu', 'mon', 'wed', 'tue', 'fri'] = Field(min_length=1,max_length=50)
    duration       :   float = Field(ge=0)
    campaign       :   float
    pdays          :   float
    previous       :   float
    poutcome       :   Literal['nonexistent', 'failure', 'success'] = Field(min_length=1,max_length=50)
    emp_var_rate   :   float = Field(alias='emp.var.rate')
    cons_price_idx :   float = Field(alias='cons.price.idx')
    cons_conf_idx  :   float = Field(alias='cons.conf.idx')
    euribor3m      :   float
    nr_employed    :   float = Field(alias='nr.employed')

class ResponseModel(BaseModel):
    prediction : int
    probability : float


@app.get("/")
def home():
    return {"message": "API is running"}

@app.post('/predict',response_model=ResponseModel)
def get_input(data:InputModel, artifact:dict = Depends(get_artifact)):
    raw_df = pd.DataFrame([data.model_dump(by_alias=True)])
    try:
        cleaned_df = clean(raw_df)
        pred_df = predict_new(cleaned_df,artifact)
        return {'prediction':pred_df['prediction'].iloc[0],'probability':(pred_df['probability'].iloc[0]).item()}

    except DataCleaningError as e:
        raise HTTPException(status_code=400, detail='Bad Request Error')
    
    except PredictionError as e:
        raise HTTPException(status_code=500, detail='Internal Server Error')
    
    except Exception as e:
        logger.error('Unhandled error during prediction: %s', str(e))
        raise HTTPException(status_code=500, detail='Internal Server Error')