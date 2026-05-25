import numpy as np
from sklearn.feature_selection import VarianceThreshold


def columna_mayor_varianza(df):
    df_num = df.select_dtypes(include=[np.number])

    selector = VarianceThreshold()
    selector.fit(df_num)

    varianzas = df_num.var()
    col_max = varianzas.idxmax()

    return col_max
