#Pregunta: limpiar_outliers

#Enunciado

En el análisis de datos de exoplanetas, es común encontrar mediciones erróneas o valores extremos (outliers) debido a errores de instrumentación o ruido en las observaciones.

Se dispone de un DataFrame con variables numéricas como masa, radio, temperatura estelar y distancia orbital. Además, algunos valores pueden estar ausentes (NaN).

Antes de entrenar modelos de machine learning, es necesario limpiar estos datos eliminando observaciones atípicas que puedan afectar el rendimiento del modelo.

#Función a implementar

Escribe una función llamada limpiar_outliers(df, target_col) que realice lo siguiente:

- Separe las características (X) de la variable objetivo (y).
- Impute los valores faltantes en X usando el promedio de cada columna.
- Calcule el z-score de cada valor y elimine las filas donde cualquier característica tenga un valor absoluto mayor a 3.
- Devuelva dos arrays de numpy: la matriz X limpia y el vector y correspondiente.

#Restricciones

- La columna objetivo no debe ser transformada.
- Deben usarse pandas, numpy y sklearn.
- La imputación debe hacerse con strategy="mean".
- No usar loops explícitos para eliminar outliers.

#Ejemplo

import pandas as pd
import numpy as np

df = pd.DataFrame({
    "planet_radius": [1.2, 100, 2.5],
    "planet_mass": [5.1, 200, 3.2],
    "star_temperature": [5000, 6000, 5500],
    "is_habitable": [1, 0, 1]
})

X, y = limpiar_outliers(df, "is_habitable")
