from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,classification_report
import os
from dotenv import load_dotenv
from src.data.cleaning import clean
from src.workflows.pipeline import build_preprocessor,build_model_pipeline
import random
import pandas as pd
import numpy as np
import joblib
# Global seeds
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)



def load_and_split_data(dataset_path:str, target_col:str = 'y',test_split:float = 0.2):
    df = pd.read_csv(dataset_path,delimiter=';')
    df = clean(df)
    X = df.drop(columns=[target_col])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_split,random_state=42,stratify=y)
    return  X_train, X_test, y_train, y_test, encoder

def get_column_groups(X:pd.DataFrame, exclude_cols:list[str]):
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(include=['str','object']).columns.difference(exclude_cols)
    return num_cols,cat_cols

def build_model(preprocessor:ColumnTransformer,y_train) -> dict[str,Pipeline]:
    models = {}
    estimator = DecisionTreeClassifier(max_depth=3,random_state=42,min_samples_leaf=4,min_samples_split=3)
    weight_ratio = (y_train == 0).sum()/(y_train == 1).sum()

    models['log_reg'] = build_model_pipeline(preprocessor,LogisticRegression(class_weight='balanced',
                                                                    random_state=42,
                                                                    max_iter=1000))

    models['random_forest'] = build_model_pipeline(preprocessor,RandomForestClassifier(max_depth=5,
                                                                n_estimators=200,
                                                                class_weight='balanced',
                                                                random_state=42,n_jobs=-1))

    models['gradient_boost'] = build_model_pipeline(preprocessor,GradientBoostingClassifier(n_estimators=100,
                                                                        max_depth=3,
                                                                        random_state=42,
                                                                        learning_rate=5e-3))

    models['adaboost'] = build_model_pipeline(preprocessor,AdaBoostClassifier(estimator=estimator,
                                                                n_estimators=200,
                                                                random_state=42,
                                                                learning_rate=5e-3))

    models['xgboost'] = build_model_pipeline(preprocessor,XGBClassifier(scale_pos_weight = weight_ratio,
                                                n_estimators = 300,
                                                learning_rate = 0.05,
                                                max_depth = 3,
                                                subsample = 0.8,
                                                reg_alpha = 1,
                                                reg_lambda = 1,
                                                colsample_bytree = 0.8,
                                                random_state = 34,
                                                objective = 'binary:logistic',
                                                eval_metric = 'logloss'
                                            ))
    return models
    
def cross_val_for_best_model(X_train, y_train, models:dict[str,Pipeline]) -> Pipeline:
    record = []
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    for model in models.values():
        score = []
        for train_idx,val_idx in skf.split(X_train,y_train):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train[train_idx]

            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train[val_idx]

            model.fit(X_train_fold,y_train_fold)
            y_pred_prob = model.predict_proba(X_val_fold)[:,1]
            score.append(roc_auc_score(y_val_fold,y_pred_prob))

        record.append({'model' : model,'model_name' : type(model.named_steps['clf']).__name__, 'score' : np.mean(score)})
    df_score = pd.DataFrame(data=record)
    best_row = df_score.iloc[df_score['score'].argmax()]

    return best_row['model']

def train_best_model(X_train,y_train,best_model):
    best_model.fit(X_train,y_train)
    return best_model

def evaluate_model(X_test,y_test,model : Pipeline):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test,y_pred))
    print('ROC-AUC score',roc_auc_score(y_test,y_prob))

def main():
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')
    X_train,X_test,y_train,y_test,encoder = load_and_split_data(DATASET_PATH,'y',0.2)
    num_cols,cat_cols = get_column_groups(X_train,exclude_cols=['default'])
    preprocessor = build_preprocessor(num_cols,cat_cols,['default'])

    models = build_model(preprocessor,y_train)
    best_model= cross_val_for_best_model(X_train,y_train,models)
    trained_model = train_best_model(X_train,y_train,best_model)
    return X_test,y_test,trained_model,encoder


if __name__ == '__main__':
    X_test,y_test,final_model,encoder = main()
    evaluate_model(X_test,y_test,final_model)
    os.makedirs('artifact', exist_ok=True)
    joblib.dump({'model':final_model,'encoder':encoder},'artifact/artifacts.joblib')

