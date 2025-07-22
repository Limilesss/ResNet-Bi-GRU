# ResNet-Bi-GRU

A modular deep learning framework for time-series classification using a ResNet + Bi-GRU hybrid architecture. This project is designed for electrical signal fault diagnosis using processed current and voltage time-series data.

## 📁 Project Structure

```
ResNet-Bi-GRU/
├── data_loader/
├── preprocessing/
├── features/
├── models/
├── training/
├── evaluation/
├── visualization/
├── main.py
├── README.md
└── requirements.txt
```

## 🚀 Features

- Residual CNN blocks for spatial feature extraction
- Bidirectional GRU for temporal dependencies
- PCA-based dimensionality reduction
- Sliding window for sequential modeling
- Cosine annealing learning rate scheduler
- Training & evaluation visualization

## 🧪 Usage

1. Place your CSV time-series data in a folder and update the path in `main.py`.

2. Run the training pipeline:

```bash
python main.py
```

## 🛠️ Dependencies

See `requirements.txt`.

## 📊 Input CSV Columns

The model expects the following input columns:

- `IsLa [A]`, `IsLb [A]`, `IsLc [A]`
- `UsLLa [V]`, `UsLLb [V]`, `UsLLc [V]`

Each row corresponds to a time point.

## 📜 License

MIT License. Feel free to use and adapt.

## 🛠️ Environment Setup

### Python Version
This project requires **Python 3.8+**

### Install Dependencies

First, make sure you have Python and `pip` installed. Then install all required packages using:

```bash
pip install -r requirements.txt
```