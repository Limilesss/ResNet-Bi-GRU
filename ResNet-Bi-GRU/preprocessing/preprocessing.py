import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def add_gaussian_noise_snr(data, snr_db):
    signal_power = np.mean(data ** 2, axis=0)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def preprocess_group_with_pca(group, label, window_size=10, snr_db=0, pca_components=6):
    features = group[['IsLa [A]', 'IsLb [A]', 'IsLc [A]',
                      'UsLLa [V]', 'UsLLb [V]', 'UsLLc [V]',
                      'Current_Product', 'Voltage_Product']].values
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    features_noisy = add_gaussian_noise_snr(features_standardized, snr_db)
    diff = np.diff(features_noisy, axis=0)
    diff = np.vstack([diff, diff[-1]])

    if diff.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        diff = pca.fit_transform(diff)

    X_group, Y_group = [], []
    for i in range(window_size, len(diff)):
        window = diff[i - window_size:i].flatten()
        X_group.append(window)
        Y_group.append(label)

    return X_group, Y_group
