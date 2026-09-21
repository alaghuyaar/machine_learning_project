import numpy as np
import pandas as pd
import logging
from src.exceptions import DataCleaningError

logger = logging.getLogger(__name__)


def lower_categorical_cols(df : pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object','str']).columns
    df[cat_cols] =  df[cat_cols].apply(lambda x : x.str.lower())
    return df

def replace_unknowns(df : pd.DataFrame,exclude_cols : list[str]|None = None) -> pd.DataFrame:
    df = df.copy()
    unknown_vals = ['unknown','?']
    req_col = df.columns.difference(exclude_cols)
    df[req_col] = df[req_col].replace(unknown_vals,np.nan)
    return df

def drop_fully_null_rows(df : pd.DataFrame) -> pd.DataFrame:
    result = df.dropna(how='all').reset_index(drop=True)
    if len(result) < len(df):
        logger.info('Dropped %d fully-null rows', len(df) - len(result))
    return result


def engineer_contacted_before(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['contacted_before'] = (df['pdays'] != 999).astype(int)
    return df

def drop_leaky_cols(df : pd.DataFrame, cols = ('duration',)) -> pd.DataFrame:
    return df.drop(columns=list(cols))
    

def clean(df : pd.DataFrame) -> pd.DataFrame:
    logger.info('Raw dataframe received : shape=%s', df.shape)
    try:
        df = lower_categorical_cols(df)
        df = replace_unknowns(df,['default'])
        df = drop_fully_null_rows(df)
        df = engineer_contacted_before(df)
        df = drop_leaky_cols(df)

        logger.info('DataFrame Cleaned : shape=%s',df.shape)
        return df.reset_index(drop=True)
    
    except KeyError as e:
        col = e.args[0]
        logger.error('Missing Column Error : %s',col)
        raise DataCleaningError('Failed to clean the Data',detail=f'Required Column "{col}" Missing !!') from e
    
    except Exception as e:
        logger.error('Encountered error : %s',str(e))
        raise DataCleaningError('Failed to clean the Data') from e

