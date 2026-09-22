from src.workflows.pipeline import build_model_pipeline,build_preprocessor
import numpy as np
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

BASE_CLEANED_COLUMN = {'age': [45,46,47,45],'job': ['admin', 'blue-collar', np.nan, 'services'],'marital': ['married', np.nan, 'divorced', np.nan],'education': [np.nan, 'high.school', 'basic.9y','basic.4y'],
    'default': ['no','yes','unknown','yes'],'housing': ['no','yes',np.nan,'yes'],'loan': ['no','yes',np.nan,'yes'],'contact': ['telephone','cellular','telephone','telephone'],
    'month': ['may','sep', 'mar', 'dec'],'day_of_week': ['mon', 'wed', 'tue', 'fri'],'campaign': [1.0,3,4.5,1],'pdays': [999,2,999,1],'previous': [0.0,0,0,1],
    'poutcome': ['nonexistent','nonexistent', 'failure', 'success'],'emp.var.rate': [1.1,2.2,3.3,4.4],'cons.price.idx': [93.994,45,56,78],'cons.conf.idx': [-36.4,45,34,23],
    'euribor3m': [4.857,4,5,6],'nr.employed': [5191.0,455,6777,7777],'y': ['no','yes','no','no'],'contacted_before':[0,1,0,1]}

@pytest.fixture
def make_df():
    data = {**BASE_CLEANED_COLUMN}
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def get_col(make_df):
    df = make_df #here make_df = df (sicne we returning a Dataframe) and not a function so its' not callable
    num_cols = df.select_dtypes(include=[int,float]).columns
    cat_cols = df.select_dtypes(include=[str,object]).columns.difference(['default','y'])
    return (num_cols,cat_cols)


def test_build_preprocessor(make_df,get_col):
    df = make_df
    num_cols,cat_cols = get_col
    preprocessor = build_preprocessor(num_cols,cat_cols,custom_cols=['default'])
    transformed = preprocessor.fit_transform(df) #column tansformer already tranform to the dense matrix
    assert not np.isnan(transformed).any()

def test_build_model_pipeline(get_col):
    num_cols,cat_cols = get_col
    preprocessor = build_preprocessor(num_cols,cat_cols,custom_cols=['default'])
    model_a = build_model_pipeline(preprocessor, LogisticRegression())
    model_b = build_model_pipeline(preprocessor, LogisticRegression())

    assert model_a.named_steps['preprocessing'] is not model_b.named_steps['preprocessing']
    assert model_a.named_steps['preprocessing'] is not preprocessor

    assert list(model_a.named_steps.keys()) == ['preprocessing','clf']
    assert isinstance(model_a.named_steps['clf'], LogisticRegression)
    