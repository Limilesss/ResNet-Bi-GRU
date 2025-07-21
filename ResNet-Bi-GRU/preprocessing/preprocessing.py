import numpy as np
import pandas as pd

def clean_and_label_data(data):
    data.fillna(data.mean(numeric_only=True), inplace=True)
    numeric_cols = data.select_dtypes(include=[np.number])

    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1

    data = data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

    label_mapping = {src: idx for idx, src in enumerate(data['Source'].unique())}
    data['Label'] = data['Source'].map(label_mapping)

    return data, label_mapping