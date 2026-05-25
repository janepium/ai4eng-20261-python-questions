import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluar_regularizacion_ridge(df, target_col, alphas):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.select_dtypes(include=[np.number])

    resultados = {}

    for alpha in alphas:
        modelo = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

        scores = cross_val_score(
            modelo,
            X,
            y,
            cv=5,
            scoring="neg_mean_squared_error"
        )

        rmse_promedio = float(np.mean(np.sqrt(-scores)))
        resultados[alpha] = rmse_promedio

    mejor_alpha = min(resultados, key=resultados.get)
    mejor_rmse = resultados[mejor_alpha]

    return resultados, mejor_alpha, mejor_rmse
