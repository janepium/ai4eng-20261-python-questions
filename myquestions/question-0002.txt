El Problema:

Recibes un DataFrame de pandas con datos numéricos de exoplanetas, donde algunas columnas tienen valores faltantes (NaN) y escalas muy diferentes. En lugar de usar estandarización, deseas escalar los datos a un rango específico [0, 1] para un modelo que requiere valores acotados.

Tu Misión: Escribe una función llamada escalar_minmax(df, target_col) que realice lo siguiente:

Separe las características (X) de la variable objetivo (y, especificada por target_col).

Impute los valores faltantes en X usando el promedio de cada columna (utilizando SimpleImputer de sklearn).

Escale las características imputadas al rango [0, 1] utilizando MinMaxScaler de sklearn.

Devuelva dos arrays de numpy: la matriz X procesada y el vector y.
