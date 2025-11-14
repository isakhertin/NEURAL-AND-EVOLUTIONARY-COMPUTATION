import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pathlib import Path


from NeuralNet import NeuralNet
from BPF import BP_F


# Setting correct path to datafile
base = Path(__file__).resolve().parents[1]  
data_file = base / "data" / "raw" / "insurance.csv"



def load_dataset():
    # SOURCE OF DATASET = https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset
    df = pd.read_csv(data_file)
    #print(df.head())
    #print(df.info())
    #print(df.describe())

    #We need 10 --> featuring engineering can help us :) 
    df["bmi_squared"] = df["bmi"]**2
    df["age_bmi"] = df["age"] * df["bmi"]
    df["is_child"] = (df["age"] < 18).astype(int)
    df["smoker_bmi"] = df["bmi"] * (df["smoker"] == "yes").astype(int)

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
    y_train = y_train_scaled * target_std + target_mean

    print("Missed values:")
    print(df.isna().sum())
    print("\nOutliers:")   
    print(df.describe(),"\n")     

    return X_train, X_test, y_train_scaled, y_train, y_test, target_mean, target_std


def train_mlr(X_train, y_train, X_test):
    lin = LinearRegression()
    lin.fit(X_train, y_train)       # OBS! y_train i originalskala
    y_pred_mlr = lin.predict(X_test)
    return y_pred_mlr



def get_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def get_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    # Part 1
    (
        X_train,
        X_test,
        y_train_scaled,
        y_train,
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
    y_pred_bp = y_pred_scaled * target_std + target_mean

    # Evaluate
    mse_bp = get_mse(y_test, y_pred_bp)
    mae_bp = get_mae(y_test,y_pred_bp)
    mape_bp = get_mape(y_test,y_pred_bp)

    print(f"\nTest Results:")
    print(f"MSE: {mse_bp:.2f}")
    print(f"MAE: {mae_bp:.2f}")
    print(f"MAPE: {mape_bp:.2f}%")

    #Train MLR
    print("\nvalutation of MLR")
    y_pred_mlr = train_mlr(X_train, y_train, X_test)
    mse_mlr = get_mse(y_test,y_pred_mlr)
    mae_mlr = get_mae(y_test,y_pred_mlr)
    mape_mlr = get_mape(y_test,y_pred_mlr)
    print(f"MSE: {mse_mlr:.2f}")
    print(f"MAE: {mae_mlr:.2f}")
    print(f"MAPE: {mape_mlr:.2f}%")


    #Train BPF
    print("\nTraining BPF..")
    bpf = BP_F(layers)
    bpf.fit(
        X_train=X_train,
        y_train=y_train_scaled,    
        epochs=800,
        lr=0.01,
    )

    # Predict on the test set
    y_pred_bpf_scaled = bpf.predict(X_test)
    y_pred_bpf = y_pred_bpf_scaled * target_std + target_mean
    mse_bpf = get_mse(y_test,y_pred_bpf)
    mae_bpf = get_mae(y_test,y_pred_bpf)
    mape_bpf = get_mape(y_test,y_pred_bpf)

    print("\nBP-F Test Results:")
    print(f"MSE: {mse_bpf:.2f}")
    print(f"MAE: {mae_bpf:.2f}")
    print(f"MAPE: {mape_bpf:.2f}%")
    
    
    #Result print 
    results = pd.DataFrame({
    "Model": ["BP (egen)", "MLR-F", "BP-F (PyTorch)"],
    "MSE": [mse_bp, mse_mlr, mse_bpf],
    "MAE": [mae_bp, mae_mlr, mae_bpf],
    "MAPE": [mape_bp, mape_mlr, mape_bpf]
    })
    print(results)


    #PLOT
    # BP
    plt.subplot(1,3,1)
    plt.scatter(y_test, y_pred_bp)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title("BP")

    # MLR-F
    plt.subplot(1,3,2)
    plt.scatter(y_test, y_pred_mlr)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title("MLR-F")

    # BP-F
    plt.subplot(1,3,3)
    plt.scatter(y_test, y_pred_bpf)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title("BP-F")

    plt.show()


if __name__ == "__main__":
    main()
