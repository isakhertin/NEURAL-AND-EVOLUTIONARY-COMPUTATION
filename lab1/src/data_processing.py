from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

base = Path(__file__).resolve().parents[1]  
data_file = base / "data" / "raw" / "insurance.csv"



def load_dataset():
    df = pd.read_csv(data_file)

    #We need 10 --> featuring engineering can help us :) 
    df["bmi_squared"] = df["bmi"]**2
    df["age_bmi"] = df["age"] * df["bmi"]
    df["is_child"] = (df["age"] < 18).astype(int)
    df["smoker_bmi"] = df["bmi"] * (df["smoker"] == "yes").astype(int)

    # Separetae features and target
    X = df.drop(columns=['charges'])
    y = df['charges']


    X = pd.get_dummies(X, drop_first=True)
    
    # Split befor normalizing to prevent data leakage
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
