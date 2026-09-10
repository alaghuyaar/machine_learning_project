from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator,clone


def build_preprocessor(num_cols : list[str], cat_cols : list[str], custom_cols : list[str]) -> ColumnTransformer:

    num_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='mean')),
        ('scale',StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder())
    ])

    custom_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='unknown')),
        ('encode',OneHotEncoder())
    ])


    preprocessor = ColumnTransformer([
        ('num_col',num_pipeline,num_cols),
        ('cat_col',cat_pipeline,cat_cols),
        ('custom_col',custom_pipeline,custom_cols)
    ])

    return preprocessor

def build_model_pipeline(preprocessor: ColumnTransformer,estimator : BaseEstimator) -> Pipeline:

    model = Pipeline([
        ('preprocessing',clone(preprocessor)),
        ('clf',estimator)
    ])
    return model
