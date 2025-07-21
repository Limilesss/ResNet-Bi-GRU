import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, Y_test):
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return cm, acc, precision, recall, f1