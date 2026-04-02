import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import random

def generar_caso_de_uso_escalar_minmax():

    n_rows = random.randint(5, 15)
    n_features = random.randint(2, 5)

    data = np.random.randn(n_rows, n_features) * random.randint(1, 100)
    feature_cols = [f'feature_{i}' for i in range(n_features)]

    df = pd.DataFrame(data, columns=feature_cols)

    mask = np.random.choice([True, False], size=df.shape, p=[0.15, 0.85])
    df[mask] = np.nan

    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    output_data = (X_scaled, y)

    return input_data, output_data
