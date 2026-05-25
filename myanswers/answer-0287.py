import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reducir_dimensiones_sensores(df_lecturas):
    df_limpio = df_lecturas.dropna()

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    pca = PCA(n_components=2)
    resultado_pca = pca.fit_transform(datos_escalados)

    return resultado_pca
