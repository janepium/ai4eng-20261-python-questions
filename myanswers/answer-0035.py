import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve


def analizar_curva_aprendizaje(X, y, modelo=None, cv=5):
    """
    Analiza una curva de aprendizaje usando learning_curve de sklearn.

    Retorna un diccionario con:
    - train_mean: promedio de los scores de entrenamiento por tamaño de muestra
    - val_mean: promedio de los scores de validación por tamaño de muestra
    - diagnostico: diagnóstico del ajuste del modelo
    """

    if modelo is None:
        modelo = RandomForestClassifier(random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        modelo,
        X,
        y,
        cv=cv
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    if train_mean[-1] > val_mean[-1] + 0.1:
        diagnostico = "Posible sobreajuste"
    elif train_mean[-1] < 0.6:
        diagnostico = "Posible subajuste"
    else:
        diagnostico = "Buen ajuste"

    return {
        "train_mean": train_mean,
        "val_mean": val_mean,
        "diagnostico": diagnostico
    }
