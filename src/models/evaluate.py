from sklearn.pipeline import Pipeline
from .train import main
from sklearn.metrics import classification_report,roc_auc_score




def evaluate_model(X_test,y_test,model : Pipeline):
    y_pred = model.predict(X_test)         
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", auc)

    return {'y_pred': y_pred, 'y_proba': y_proba, 'roc_auc': auc}




X_test,y_test,model = main()
evaluate_model(X_test,y_test,model)