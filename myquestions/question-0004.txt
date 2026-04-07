# Pregunta: normalizar_l2

## Enunciado

En ciertos modelos de machine learning, como K-Nearest Neighbors o algoritmos de clustering, es importante que cada observación tenga la misma magnitud para evitar que algunas dominen el cálculo de distancias.

Se dispone de un DataFrame con características numéricas de exoplanetas, como masa, radio, temperatura estelar y distancia orbital. Sin embargo, algunos valores están incompletos (NaN).

Se requiere normalizar cada fila del dataset para que tenga norma L2 igual a 1, lo que permite comparar observaciones de manera más justa.

## Función a implementar

Escribe una función llamada normalizar_l2(df, target_col) que realice lo siguiente:

- Separe las características (X) de la variable objetivo (y).
- Impute los valores faltantes en X usando el promedio de cada columna.
- Normalice cada fila de X para que tenga norma L2 igual a 1, utilizando normalize de sklearn.preprocessing.
- Devuelva dos arrays de numpy: la matriz X normalizada y el vector y.

## Restricciones

- La columna objetivo no debe ser transformada.
- Deben usarse pandas, numpy y sklearn.
- La imputación debe hacerse con strategy="mean".
- La normalización debe realizarse con norma L2.

## Ejemplo

import pandas as pd
import numpy as np

df = pd.DataFrame({
    "planet_radius": [1.2, np.nan, 2.5],
    "planet_mass": [5.1, 3.2, np.nan],
    "distance_from_star": [1.0, 0.8, 1.2],
    "is_habitable": [1, 0, 1]
})

X, y = normalizar_l2(df, "is_habitable")
