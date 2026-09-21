from src.data.cleaning import lower_categorical_cols,replace_unknowns,drop_fully_null_rows,engineer_contacted_before,drop_leaky_cols
from src.data.cleaning import clean
from src.exceptions import DataCleaningError
import pandas as pd
import numpy as np
import pytest

#this doesn't have duration col and totally fixed
BASE_COLUMN = {'age': [45,46,47,45],'job': ['Admin', 'Blue-Collar', 'UNKNOwn', 'Services'],'marital': ['married', 'unknown', 'divorced', np.nan],'education': [np.nan, 'high.school', 'basic.9y','basic.4y'],'duration':[20.0,45,3,4],
    'default': ['no','yes','unknown','yes'],'housing': ['no','yes',np.nan,'yes'],'loan': ['no','yes',np.nan,'yes'],'contact': ['telephone','cellular','telephone','telephone'],
    'month': ['may','sep', 'mar', 'dec'],'day_of_week': ['mon', 'wed', 'tue', 'fri'],'campaign': [1.0,3,4.5,1],'pdays': [999,2,999,1],'previous': [0.0,0,0,1],
    'poutcome': ['nonexistent','nonexistent', 'failure', 'success'],'emp.var.rate': [1.1,2.2,3.3,4.4],'cons.price.idx': [93.994,45,56,78],'cons.conf.idx': [-36.4,45,34,23],
    'euribor3m': [4.857,4,5,6],'nr.employed': [5191.0,455,6777,7777],'y': ['no','yes','no','no']}


@pytest.fixture
def make_df():
    def _make(**overide):
        data = {**BASE_COLUMN,**overide}
        df = pd.DataFrame(data)
        return df
    return _make
def test_lower_categorical_cols(make_df):
    raw_df = make_df()
    result = lower_categorical_cols(raw_df)

    expected_df = make_df(job=['admin', 'blue-collar', 'unknown', 'services'])
    assert result.equals(expected_df)


def test_replace_unknowns(make_df):
    raw_df = make_df(job= ['admin', 'blue-collar', 'unknown', 'services'],marital= ['married', 'unknown', 'divorced', np.nan])
    result = replace_unknowns(raw_df,['default'])

    expected_df = make_df(job= ['admin', 'blue-collar', np.nan, 'services'],marital= ['married', np.nan, 'divorced', np.nan])
    assert result.equals(expected_df)


def test_drop_fully_null_rows(make_df):
    raw_df = make_df()
    raw_df.iloc[len(raw_df) - 1] = np.nan

    result = drop_fully_null_rows(raw_df)
    expected_df = raw_df.dropna(how='all').reset_index(drop=True)

    assert result.reset_index(drop=True).equals(expected_df)


def test_engineer_contacted_beforee(make_df):
    raw_df = make_df()
    result = engineer_contacted_before(raw_df)

    expected_df = make_df(contacted_before=[0,1,0,1])
    assert result.equals(expected_df)


def test_drop_leaky_cols(make_df):
    raw_df = make_df()
    result = drop_leaky_cols(raw_df)

    expected_df = make_df()
    assert result.equals(expected_df.drop(columns=['duration']))


#testing the clean pipeline : here df11 is a raw df with Uppercase values,'unknown',no 'contacted_before' and 'duration' col
def test_clean_pipeline(make_df):
    raw_df = make_df()
    result = clean(raw_df)

    expected_df = make_df(job= ['admin', 'blue-collar', np.nan, 'services'],marital= ['married', np.nan, 'divorced', np.nan],contacted_before=[0,1,0,1])
    assert 'duration' not in result.columns        
    assert 'contacted_before' in result.columns
    assert result.equals(expected_df.drop(columns=['duration']))



#testing the exception of clean
def test_clean_Exception(monkeypatch,make_df):
    def broken_step(df):
        raise ValueError('Boom')

    monkeypatch.setattr('src.data.cleaning.lower_categorical_cols',broken_step)
    raw_df = make_df()
    with pytest.raises(DataCleaningError) as exc_info:
        clean(raw_df)
    assert str(exc_info.value) == "Failed to clean the Data"
    assert exc_info.value.detail == None

def test_clean_Keyerror_Exception(monkeypatch,make_df):
    def broken_step(df):
        raise KeyError('OOOPS')

    monkeypatch.setattr('src.data.cleaning.drop_leaky_cols',broken_step)
    raw_df = make_df()
    with pytest.raises(DataCleaningError) as exc_info:
        clean(raw_df)

    assert str(exc_info.value) == "Failed to clean the Data"
    assert exc_info.value.detail == 'Required Column "OOOPS" Missing !!'
