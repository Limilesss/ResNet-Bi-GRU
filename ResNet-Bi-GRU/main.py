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

     # 2. 缺失值 + 异常值清洗
    raw_data.fillna(raw_data.mean(numeric_only=True), inplace=True)
    numeric_columns = raw_data.select_dtypes(include=[np.number])
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    raw_data = raw_data[~((numeric_columns < (Q1 - 1.5 * IQR)) |
                          (numeric_columns > (Q3 + 1.5 * IQR))).any(axis=1)]

    # 3. 标签编码 + 特征构造
    label_mapping = {src: idx for idx, src in enumerate(raw_data['Source'].unique())}
    raw_data['Label'] = raw_data['Source'].map(label_mapping)
    raw_data['Current_Product'] = raw_data['IsLa [A]'] * raw_data['IsLb [A]'] * raw_data['IsLc [A]']
    raw_data['Voltage_Product'] = raw_data['UsLLa [V]'] * raw_data['UsLLb [V]'] * raw_data['UsLLc [V]']

    # 4. 滑动窗口 + SNR + 差分 + PCA
    window_size = 10
    snr_db = 0
    pca_components = 6
    X_all, Y_all = [], []

    for source, group in raw_data.groupby('Source'):
        if len(group) > window_size:
            label = label_mapping[source]
            X_group, Y_group = preprocess_group_with_pca(group, label, 
                                                         window_size=window_size, 
                                                         snr_db=snr_db, 
                                                         pca_components=pca_components)
            X_all.extend(X_group)
            Y_all.extend(Y_group)

    X = np.array(X_all)
    onehot = OneHotEncoder(sparse_output=False)
    Y = onehot.fit_transform(np.array(Y_all).reshape(-1, 1))

    # 5. 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")
    print(f"X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")

    # 6. 构建并训练模型
    input_shape = (X_train.shape[1], 1)
    model = build_resnet_gru_model(input_shape=input_shape, num_classes=Y_train.shape[1])
    model.compile(optimizer=Nadam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    scheduler = CosineAnnealingScheduler(T_max=150, eta_min=1e-6)
    history = model.fit(
        X_train.reshape(-1, input_shape[0], 1), Y_train,
        validation_data=(X_val.reshape(-1, input_shape[0], 1), Y_val),
        epochs=150, batch_size=64,
        callbacks=[scheduler]
    )

    # 7. 模型评估
    cm, acc, prec, rec, f1 = evaluate_model(model, X_test.reshape(-1, input_shape[0], 1), Y_test)
    print(f"\n[Test Results]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 8. 可视化结果
    plot_history(history.history)
    plot_confusion_matrix(cm)

if __name__ == '__main__':
    main()
