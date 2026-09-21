from fastapi.testclient import TestClient
from src.api.main import get_artifact,app
import pytest
from unittest.mock import Mock
import numpy as np
from src.exceptions import DataCleaningError, PredictionError


def fake_artifact():
    fake_model = Mock()
    #fixing values for model
    fake_model.predict.return_value = np.array([0]) #value we get when artifact['model'] is called
    fake_model.predict_proba.return_value = np.array([[0.88,0.23]])

    fake_encoder = Mock()
    #fixing value for encoder
    fake_encoder.inverse_transform.return_value = np.array(['no'])
    return {'model':fake_model,'encoder':fake_encoder}

app.dependency_overrides[get_artifact] = fake_artifact
client = TestClient(app)

@pytest.mark.parametrize("endpoint,expected_status,expected_body",
[
    ('/',200,{'message': "API is running"})
])

def test_get(endpoint,expected_status,expected_body):
    response = client.get(endpoint)

    assert response.status_code == expected_status
    assert response.json() == expected_body


@pytest.mark.parametrize("expected_status,expected_col,payload",
[   #correct status code and payload
    (200,None,{'age': 46,'job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0}),
    #correct status code but field 'marital' with incorrect value. Should be 'unmarried' or 'married'
    (422,'marital',{'age': 46,'job': 'services','marital': 'non.married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0}),
    #correct status code but field 'age' with incorrect data type
    (422,'age',{'age': 'forty.six','job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0}),
    # #correct status code and field 'age' with value in incorrect range
    (422,'age',{'age': -1,'job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0})
        
])

def test_post(expected_status,expected_col,payload):
    response = client.post('/predict',json=payload)
    assert response.status_code == expected_status
    if response.status_code == 200:
        body = response.json()
        assert 'prediction' in body 
        assert body['prediction'] in [0,1]
        assert 0.0 <= body['probability'] <= 1.0

    elif response.status_code == 422:
        body = response.json()
        assert 'detail' in body
        assert any([expected_col in err.get('loc') for err in body['detail']])

    elif response.status_code == 500:
        body = response.json()
        assert 'detail' in body
        assert body['detail'] == 'Internal Server Error'
        assert 'Traceback' not in body['detail']

    elif response.status_code == 400:
        body = response.json()
        assert 'detail' in body
        assert body['detail'] == 'Bad Request Error'


@pytest.mark.parametrize("expected_status,df",[(400,{'age': 34,'job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0})])
def test_400_post_DataCleaningError(monkeypatch,expected_status,df):
    def broken_clean(df):
        raise DataCleaningError('Failed to clean the Data') 
    
    monkeypatch.setattr('src.api.main.clean',broken_clean)
    response = client.post('/predict',json=df)
    body = response.json()
    assert response.status_code == expected_status
    assert body['detail'] ==  'Bad Request Error'


def test_500_post_PredictionError(monkeypatch):
    def broken_predict(df,artifact):
        raise PredictionError('Prediction failed')

    monkeypatch.setattr('src.api.main.predict_new',broken_predict)

    df  = {'age': 34,'job': 'services','marital': 'married','education': 'high.school','duration':20.0,
            'default': 'no','housing': 'no','loan': 'no','contact': 'telephone',
            'month': 'may','day_of_week': 'mon','campaign': 1.0,'pdays': 999.0,'previous': 0.0,
            'poutcome': 'nonexistent','emp.var.rate': 1.1,'cons.price.idx': 93.994,'cons.conf.idx': -36.4,
            'euribor3m': 4.857,'nr.employed': 5191.0}
    response = client.post('/predict',json=df)
    body = response.json()
    assert response.status_code == 500
    assert body['detail'] ==  'Internal Server Error'

