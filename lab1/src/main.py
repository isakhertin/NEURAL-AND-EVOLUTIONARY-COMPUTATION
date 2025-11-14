import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from NeuralNet import NeuralNet


# Setting correct path to datafile
base = Path(__file__).resolve().parents[1]  
data_file = base / "data" / "raw" / "insurance.csv"



def load_dataset():
    # SOURCE OF DATASET = https://www.kaggle.com/datasets/moneystore/agencyperformance
    df = pd.read_csv(data_file)
    #print(df.head())
    #print(df.info())
    #print(df.describe())

    # Separetae features and target
    X = df.drop(columns=['charges'])
    y = df['charges']


    X = pd.get_dummies(X, drop_first=True)
    
    # Split innan normalisering för att undvika dataläckage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.values.astype(np.float64)
    y_test = y_test.values.astype(np.float64)
    target_mean = y_train.mean()
    target_std = y_train.std()
    if target_std == 0:
        target_std = 1.0
    y_train_scaled = (y_train - target_mean) / target_std

    return X_train, X_test, y_train_scaled, y_test, target_mean, target_std

def main():
    # Part 1
    (
        X_train,
        X_test,
        y_train_scaled,
        y_test,
        target_mean,
        target_std,
    ) = load_dataset()
    print(f"Träningsdata: {X_train.shape}, Testdata: {X_test.shape}")

    # Part 2 Create NN
    input_size = X_train.shape[1]
    layers = [input_size, 9, 5, 1]
    nn = NeuralNet(layers)
    
    # Train network
    nn.fit(
        X_train,
        y_train_scaled,
        epochs=800,
        lr=0.01,
        momentum=0.9,
        activation_hidden="tanh",
        activation_output="linear",
        shuffle=True,
    )

    # Predict on test data
    y_pred_scaled = nn.predict(
        X_test,
        activation_hidden="tanh",
        activation_output="linear",
    ).flatten()
    y_pred = y_pred_scaled * target_std + target_mean

    # Evaluate
    mse = np.mean((y_test - y_pred.flatten()) ** 2)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

    print(f"\nTest Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
