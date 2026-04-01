def preparar_datos(df, target_col):
    """
    Limpia y transforma un DataFrame para ML.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    target_col : str
        Nombre de la columna objetivo.

    Retorna
    -------
    X_scaled : np.ndarray
        Features imputadas y escaladas.
    y : np.ndarray
        Variable objetivo en formato numpy array.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas DataFrame")

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame")

    df = df.copy()

    # Separar features y target
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputación
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y
