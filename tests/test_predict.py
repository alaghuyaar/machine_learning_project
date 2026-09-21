import pytest
from unittest.mock import Mock
from src.models.predict import predict_new,load_artifacts
import numpy as np
import pandas as pd
from src.exceptions import ArtifactLoadError,PredictionError

BASE_COLUMN = {'age': [45,46,47,45],'job': ['Admin', 'Blue-Collar', 'UNKNOwn', 'Services'],'marital': ['married', 'unknown', 'divorced', np.nan],'education': [np.nan, 'high.school', 'basic.9y','basic.4y'],'duration':[20.0,45,3,4],
    'default': ['no','yes','unknown','yes'],'housing': ['no','yes',np.nan,'yes'],'loan': ['no','yes',np.nan,'yes'],'contact': ['telephone','cellular','telephone','telephone'],
    'month': ['may','sep', 'mar', 'dec'],'day_of_week': ['mon', 'wed', 'tue', 'fri'],'campaign': [1.0,3,4.5,1],'pdays': [999,2,999,1],'previous': [0.0,0,0,1],
    'poutcome': ['nonexistent','nonexistent', 'failure', 'success'],'emp.var.rate': [1.1,2.2,3.3,4.4],'cons.price.idx': [93.994,45,56,78],'cons.conf.idx': [-36.4,45,34,23],
    'euribor3m': [4.857,4,5,6],'nr.employed': [5191.0,455,6777,7777],'y': ['no','yes','no','no']}

@pytest.fixture
def make_df():
    def _make():
        data = {**BASE_COLUMN}
        df = pd.DataFrame(data)
        return df
    return _make


def test_predict_new(make_df):
    def fake_artifact():
        fake_model = Mock()
        fake_model.predict.return_value = np.array([0, 1])
        fake_model.predict_proba.return_value = np.array([[0.78, 0.22], [0.22, 0.78]])
        return {'model': fake_model, 'encoder': Mock()}

    df = make_df()
    artifact = fake_artifact()
    df = predict_new(df=df,artifact=artifact)
    assert (df['prediction'].iloc[0]).item() in [0,1]
    assert 0 <= (df['probability'].iloc[0]).item() <= 1


def test_predict_new_exceptions(make_df):
    fake_model = Mock()
    fake_model.predict.side_effect = ValueError('Boom')
    artifact = {'model':fake_model,'encoder':Mock()}

    df = make_df()
    with pytest.raises(PredictionError) as exec_info:
        predict_new(df,artifact)
    assert str(exec_info.value) == 'Prediction failed'

def test_load_artifacts_exceptions():
    path = 'file/not/found.joblib'
    with pytest.raises(ArtifactLoadError) as exec_info:
        load_artifacts(path)

    assert str(exec_info.value) == f'Could not load the model from {path}'