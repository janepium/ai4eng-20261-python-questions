#validacion del caso de uso y la solucion
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reducir_dimensiones_sensores(df_lecturas):
    """
    Resuelve el problema de reducción de dimensiones:
    1. Limpia nulos.
    2. Escala datos (Media 0, Varianza 1).
    3. Aplica PCA para reducir a 2 componentes.
    """
    # 1. Limpieza: Elimina filas con cualquier valor nulo
    df_limpio = df_lecturas.dropna()

    # 2. Escalamiento: StandardScaler requiere que los datos tengan media 0 y var 1
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    # 3. Reducción: Configurar PCA para 2 componentes
    pca = PCA(n_components=2)

    # 4. Transformación: Entrenar y aplicar la reducción
    resultado_pca = pca.fit_transform(datos_escalados)

    return resultado_pca

    # 1. Generamos un caso de uso aleatorio con la función anterior
input_data, output_esperado = generar_caso_de_uso_reducir_dimensiones_sensores()

# 2. Ejecutamos la función de resolución con el input generado
resultado_real = reducir_dimensiones_sensores(input_data["df_lecturas"])

# 3. Comprobamos si son iguales
son_iguales = np.allclose(resultado_real, output_esperado)

print(f"¿La función resolvió el caso correctamente?: {son_iguales}")

if son_iguales:
    print("¡Perfecto! El generador y la función están sincronizados.")
else:
    print("Hay una discrepancia en los resultados.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reducir_dimensiones_sensores(df_lecturas):
    """
    Reduce lecturas de sensores industriales a 2 componentes principales.

    Pasos:
    1. Elimina filas con valores NaN.
    2. Escala las variables con StandardScaler.
    3. Aplica PCA con 2 componentes.
    4. Retorna el arreglo transformado.
    """

    df_limpio = df_lecturas.dropna()

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    pca = PCA(n_components=2)
    resultado_pca = pca.fit_transform(datos_escalados)

    return resultado_pca
