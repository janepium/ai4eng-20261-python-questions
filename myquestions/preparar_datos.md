# Pregunta: preparar_datos

## Enunciado

En una misión espacial dedicada al estudio de exoplanetas, se recolectan mediciones astronómicas de cientos de cuerpos celestes detectados por telescopios. Cada registro representa un exoplaneta e incluye características numéricas como el periodo orbital (`orbital_period`), el radio del planeta (`planet_radius`), la temperatura de la estrella anfitriona (`star_temperature`), la distancia a su estrella (`distance_from_star`) y la excentricidad de la órbita (`eccentricity`).

Debido a limitaciones en los sensores y en las observaciones, algunos registros contienen valores faltantes (`NaN`). Además, las variables están en escalas muy diferentes: por ejemplo, la temperatura puede estar en miles de grados Kelvin, mientras que la excentricidad toma valores entre 0 y 1. Antes de entrenar un modelo de machine learning para analizar habitabilidad potencial, es necesario preparar correctamente estos datos.

## Función a implementar

Escribe una función llamada `preparar_datos(df, target_col)` que realice lo siguiente:

- Separe las características (`X`) de la variable objetivo (`y`), especificada por `target_col`.
- Impute los valores faltantes en `X` usando el promedio de cada columna, utilizando `SimpleImputer` de `sklearn`.
- Escale las características imputadas para que tengan media 0 y desviación estándar 1, utilizando `StandardScaler` de `sklearn`.
- Devuelva dos arrays de `numpy`: la matriz `X` procesada y el vector `y`.

## Restricciones

- La columna objetivo no debe ser transformada.
- Deben usarse `pandas`, `numpy` y `sklearn`.
- La imputación debe hacerse con `strategy="mean"`.
- El escalado debe realizarse con `StandardScaler`.

## Ejemplo

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "orbital_period": [100, 200, np.nan],
    "planet_radius": [1.2, np.nan, 2.5],
    "star_temperature": [5000, 6000, 5500],
    "eccentricity": [0.1, 0.3, np.nan],
    "is_potentially_habitable": [1, 0, 1]
})

X, y = preparar_datos(df, "is_potentially_habitable")
