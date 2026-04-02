import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import random

def generar_caso_de_uso_limpiar_outliers():

    n_rows = random.randint(10, 25)
    n_features = random.randint(2, 5)

    data = np.random.randn(n_rows, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]

    df = pd.DataFrame(data, columns=feature_cols)

    # meter algunos outliers
    for _ in range(random.randint(1, 3)):
        i = random.randint(0, n_rows-1)
        j = random.randint(0, n_features-1)
        df.iloc[i, j] = df.iloc[i, j] * random.randint(8, 15)

    # NaNs
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan

    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }

    X = df.drop(columns=[target_col])
    y = df[target_col]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    z_scores = ((X_imputed - X_imputed.mean()) / X_imputed.std()).abs()
    mask = (z_scores < 3).all(axis=1)

    X_clean = X_imputed[mask].to_numpy()
    y_clean = y[mask].to_numpy()

    output_data = (X_clean, y_clean)

    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_limpiar_outliers()

    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print("DataFrame (primeras 5 filas con posibles NaNs y outliers):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO (Tupla de arrays) ===")
    X_res, y_res = salida_esperada
    print(f"Shape de X limpia: {X_res.shape}")
    print(f"Shape de y: {y_res.shape}")
    print("Ejemplo de primera fila limpia:", X_res[0])
