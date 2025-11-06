import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from NeuralNet import NeuralNet


def load_dataset():
    # SOURCE OF DATASET = https://www.kaggle.com/datasets/moneystore/agencyperformance
    df = pd.read_csv("dataset/insurance.csv")
    #print(df.head())
    #print(df.info())
    #print(df.describe())

    # Separetae features and target
    X = df.drop(columns=['charges'])
    y = df['charges']


    X = pd.get_dummies(X, drop_first=True)
    
    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split på testsizes
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train.values, y_test.values

def main():
    # Part 1
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Träningsdata: {X_train.shape}, Testdata: {X_test.shape}")

    # Part 2 Create NN
    input_size = X_train.shape[1]
    layers = [input_size, 9, 5, 1]
    nn = NeuralNet(layers)
    
    # Train network
    nn.fit(X_train, y_train, epochs=500, lr=0.01, momentum=0.9, activation="tanh")

    # Predict on test data
    y_pred = nn.predict(X_test, activation="tanh")

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