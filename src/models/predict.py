import joblib
import pandas as pd
from src.data.cleaning import clean
from dotenv import load_dotenv
import os
import logging
from src.exceptions import PredictionError,DataCleaningError,ArtifactLoadError

logger = logging.getLogger(__name__)

def load_artifacts(path : str):
    logger.info('Loading Artifacts from path %s',path)
    try:
        artifact = joblib.load(path)
        logger.info('Artifact Loaded')
        return artifact
    except FileNotFoundError as e:
        logger.error('Failed to load artifact from %s: %s', path, str(e))
        raise ArtifactLoadError(f'Could not load the model from {path}') from e
    except Exception as e:
        logger.error('Failed to load artifact from %s: %s', path, str(e))
        raise ArtifactLoadError(f'Could not load the model from {path}') from e

        

def predict_new( df : pd.DataFrame, artifact : dict):
    logger.info('Running prediction on %d row(s)', len(df))
    try:
        model = artifact['model']
        encoder = artifact['encoder']
        X = df.drop(columns=['y'],errors='ignore')
        preds = model.predict(X)
        probs = model.predict_proba(X)[:,1]
        prediction = pd.DataFrame({'prediction': preds, 'probability': probs})
        logger.info('Prediction : %s',prediction)
        return prediction
    except Exception as e:
        logger.error('Failed to predict')
        raise PredictionError('Prediction failed') from e

    
def sample_case():
    data = {'age': 46,'job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0,'y': 'no'}
    
    return  pd.DataFrame(data=[data])

def main():
    try:
        load_dotenv()
        artifact_path = os.getenv('ARTIFACT_PATH')
        raw_data = sample_case()
        cleaned_df = clean(raw_data)
        artifact = load_artifacts(artifact_path)
        pred_df = predict_new(cleaned_df,artifact)
        prediction = {'prediction':int(pred_df.loc[0,'prediction']),'probability':(pred_df['probability'].iloc[0]).item()}
        print(prediction)
        

    except (ArtifactLoadError, PredictionError, DataCleaningError) as e:
        print(f'Error : {e}')
    except Exception as e:
        logger.error('Unhandled error during prediction: %s', str(e))
        print(f'Encountered error : {e}')

if __name__ == '__main__':
    main()