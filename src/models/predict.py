import joblib
import pandas as pd
from src.data.cleaning import clean
from dotenv import load_dotenv
import os

def load_artifacts(path : str):
    return joblib.load(path)

def predict_new( df : pd.DataFrame, artifact : dict):
    model = artifact['model']
    encoder = artifact['encoder']
    X = df.drop(columns=['y'],errors='ignore')
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]

    return pd.DataFrame({'prediction': preds, 'probability': probs})

    
def sample_case():
    data = {'age': 46,'job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0,'y': 'no'}
    
    return  pd.DataFrame(data=[data])

def main():
    load_dotenv()
    artifact_path = os.getenv('ARTIFACT_PATH')
    raw_data = sample_case()
    cleaned_df = clean(raw_data)
    artifact = load_artifacts(artifact_path)
    pred_df = predict_new(cleaned_df,artifact)
    print(type(artifact))
    print(pred_df)

if __name__ == '__main__':
    main()