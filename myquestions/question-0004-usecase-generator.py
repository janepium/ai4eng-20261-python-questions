import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
import random

def generar_caso_de_uso_normalizar_l2():

    n_rows = random.randint(5, 15)
    n_features = random.randint(2, 5)

    data = np.random.randn(n_rows, n_features) * 10
    feature_cols = [f'feature_{i}' for i in range(n_features)]

    df = pd.DataFrame(data, columns=feature_cols)

    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
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

    X_normalized = normalize(X_imputed, norm='l2')

    output_data = (X_normalized, y)

    return input_data, output_data
