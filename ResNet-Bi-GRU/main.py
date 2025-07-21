import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Nadam

from data_loader.loader import load_data
from preprocessing.preprocessing import clean_and_label_data
from features.feature_engineering import add_combination_features, get_processed_features
from models.resnet_bigru import build_resnet_bigru
from training.scheduler import CosineAnnealingScheduler
from training.train import train_model
from evaluation.evaluate import evaluate_model
from visualization.visualize import plot_training_history, plot_confusion_matrix

def main():
    # 1. 加载原始数据
    folder_path = 'your/data/folder/path'  # <- 修改为你的数据路径
    data = load_data(folder_path)

    # 2. 预处理
    data, label_mapping = clean_and_label_data(data)
    scaler = StandardScaler()
    data = add_combination_features(data, scaler)

    # 3. 特征工程
    X = get_processed_features(data, scaler, window_size=9)
    Y = pd.get_dummies(data['Label']).values[2:]
    Y = Y[-X.shape[0]:]

    # 4. 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # 5. 构建模型
    input_shape = (X_train.shape[1], 1)
    model = build_resnet_bigru(input_shape=input_shape, output_dim=Y_train.shape[1])
    model.compile(optimizer=Nadam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # 6. 定义调度器并训练
    scheduler = CosineAnnealingScheduler(T_max=150, eta_min=1e-6)
    history = train_model(model, X_train.reshape(-1, input_shape[0], 1), Y_train,
                          X_val.reshape(-1, input_shape[0], 1), Y_val, scheduler)

    # 7. 模型评估
    cm, acc, prec, rec, f1 = evaluate_model(model, X_test.reshape(-1, input_shape[0], 1), Y_test)
    print(f'\n[Evaluation]')
    print(f'Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}')

    # 8. 可视化结果
    plot_training_history(history)
    plot_confusion_matrix(cm)

if __name__ == '__main__':
    main()