Pregunta: escalar_minmax

Enunciado

En una misión espacial dedicada al análisis de exoplanetas, se recolectan diversas mediciones físicas como el radio del planeta (planet_radius), la masa (planet_mass), la temperatura de la estrella (star_temperature) y la distancia a la estrella (distance_from_star).

Debido a limitaciones en la recolección de datos, algunos valores están incompletos (NaN). Además, las variables se encuentran en escalas muy diferentes, lo que dificulta su uso en ciertos algoritmos de machine learning que requieren que los datos estén en un rango específico.

En este caso, se desea transformar las variables para que todas estén en el rango [0, 1], lo cual es especialmente útil para modelos sensibles a la escala.

Función a implementar

Escribe una función llamada escalar_minmax(df, target_col) que realice lo siguiente:

- Separe las características (X) de la variable objetivo (y), especificada por target_col.
- Impute los valores faltantes en X usando el promedio de cada columna, utilizando SimpleImputer de sklearn.
- Escale las características imputadas al rango [0, 1], utilizando MinMaxScaler de sklearn.
- Devuelva dos arrays de numpy: la matriz X procesada y el vector y.

Restricciones

- La columna objetivo no debe ser transformada.
- Deben usarse pandas, numpy y sklearn.
- La imputación debe hacerse con strategy="mean".
- El escalado debe realizarse con MinMaxScaler.

Ejemplo

import pandas as pd
import numpy as np

df = pd.DataFrame({
    "planet_radius": [1.2, np.nan, 2.5],
    "planet_mass": [5.1, 3.2, np.nan],
    "star_temperature": [5000, 6000, 5500],
    "distance_from_star": [1.0, 0.8, np.nan],
    "is_habitable": [1, 0, 1]
})

X, y = escalar_minmax(df, "is_habitable")
