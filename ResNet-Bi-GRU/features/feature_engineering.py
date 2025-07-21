import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def add_combination_features(data, scaler):
    data['Current_Product'] = data['IsLa [A]'] * data['IsLb [A]'] * data['IsLc [A]']
    data['Voltage_Product'] = data['UsLLa [V]'] * data['UsLLb [V]'] * data['UsLLc [V]']
    data[['Current_Product', 'Voltage_Product']] = scaler.fit_transform(data[['Current_Product', 'Voltage_Product']])
    return data

def get_processed_features(data, scaler, use_pca=True, window_size=9):
    features = data[['IsLa [A]', 'IsLb [A]', 'IsLc [A]', 'UsLLa [V]', 'UsLLb [V]', 'UsLLc [V]']]
    features_std = scaler.fit_transform(features)
    combined = np.hstack([features_std, data[['Current_Product', 'Voltage_Product']].values])

    X_diff = np.diff(combined, axis=0)
    X_diff = np.vstack([X_diff, X_diff[-1]])

    if use_pca:
        pca = PCA(n_components=6)
        X_pca = pca.fit_transform(X_diff)
    else:
        X_pca = X_diff

    return create_sliding_window_features(X_pca, window_size)

def create_sliding_window_features(data, window_size=9):
    return np.array([data[i-window_size:i].flatten() for i in range(window_size, data.shape[0])])