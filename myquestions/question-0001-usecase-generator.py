#se importa las librerías necesarias
import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso aleatorio para probar preparar_datos(df, target_col).

    Retorna
    -------
    input_data : dict
        Diccionario con:
        - df
        - target_col

    output_data : tuple
        Salida esperada de preparar_datos(input_data["df"], input_data["target_col"])
    """

    # Tamaño aleatorio
    n_rows = random.randint(6, 15)
    n_features = random.randint(3, 6)

    # Posibles columnas
    posibles_features = [
        "orbital_period",
        "planet_radius",
        "star_temperature",
        "distance_from_star",
        "stellar_mass",
        "stellar_luminosity",
        "eccentricity",
        "planet_density"
    ]

    feature_cols = random.sample(posibles_features, n_features)

    # Generar datos con escalas distintas
    data_dict = {}

    for col in feature_cols:
        if col == "orbital_period":
            values = np.random.uniform(1, 500, size=n_rows)
        elif col == "planet_radius":
            values = np.random.uniform(0.3, 15, size=n_rows)
        elif col == "star_temperature":
            values = np.random.uniform(2500, 12000, size=n_rows)
        elif col == "distance_from_star":
            values = np.random.uniform(0.01, 30, size=n_rows)
        elif col == "stellar_mass":
            values = np.random.uniform(0.1, 5, size=n_rows)
        elif col == "stellar_luminosity":
            values = np.random.uniform(0.001, 100, size=n_rows)
        elif col == "eccentricity":
            values = np.random.uniform(0, 1, size=n_rows)
        elif col == "planet_density":
            values = np.random.uniform(0.5, 15, size=n_rows)

        data_dict[col] = values

    df = pd.DataFrame(data_dict)

    # Insertar NaNs aleatorios
    prob_nan = random.uniform(0.10, 0.25)
    mask = np.random.rand(n_rows, n_features) < prob_nan
    df = df.mask(mask)

    # Columna objetivo
    target_col = "is_potentially_habitable"
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    # Input
    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    # Output esperado
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    output_data = (X_scaled, y)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_preparar_datos()

    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print("DataFrame (primeras 5 filas con posibles NaNs):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO (Tupla de arrays) ===")
    X_res, y_res = salida_esperada
    print(f"Shape de X procesada: {X_res.shape}")
    print(f"Shape de y: {y_res.shape}")
    print("Ejemplo de primera fila escalada:", X_res[0])
