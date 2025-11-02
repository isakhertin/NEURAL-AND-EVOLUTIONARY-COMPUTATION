import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset():
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

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_dataset()

print(f"Träningsdata: {X_train.shape}, Testdata: {X_test.shape}")

