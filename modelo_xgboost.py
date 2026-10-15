import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix)

#1. Funciones matematicas
def sigmoide(z):
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -z))

def perdida_logistica(y, z):
    return np.logaddexp(0, z) - y * z

def calcular_derivadas(y, z):
    p = sigmoide(z)
    gradiente = p - y
    hessiano = p * (1 - p)
    return gradiente, hessiano


#2. Funcion objetivo personalizada
def objetivo_personalizado(predicciones, dtrain):
    y = dtrain.get_label()
    gradiente, hessiano = calcular_derivadas(y, predicciones)

    pesos = dtrain.get_weight()
    if len(pesos) > 0:
        gradiente *= pesos
        hessiano *= pesos
    return gradiente, hessiano


#3. Balanceo de clases
def calcular_pesos(y):
    clases, cantidades = np.unique(y, return_counts=True)

    if len(clases) != 2:
        raise ValueError("Train debe contener ambas clases.")

    #Calcula pesos para cada clase
    pesos_clase = len(y) / (2 * cantidades)
    peso_clase_0 = pesos_clase[0]
    peso_clase_1 = pesos_clase[1]
    pesos = []

    #Asigna pesos a cada muestra según su clase
    for etiqueta in y:
        if etiqueta == clases[0]:
            pesos.append(peso_clase_0)
        else:
            pesos.append(peso_clase_1)
    return np.array(pesos)


#4. Evaluacion por participante
def evaluar(y, probabilidades):

    # Convierte probabilidades a predicciones binarias
    predicciones = []
    for probabilidad in probabilidades:
        if probabilidad >= 0.5:
            predicciones.append(1)
        else:
            predicciones.append(0)
    predicciones = np.array(predicciones)

    print("Accuracy:", accuracy_score(y, predicciones))
    print("Precision:", precision_score(y, predicciones, zero_division=0))
    print("Recall:", recall_score(y, predicciones, zero_division=0))
    print("F1:", f1_score(y, predicciones, zero_division=0))
    print("AUC:", roc_auc_score(y, probabilidades))
    print("Matriz de confusion:")
    print(confusion_matrix(y, predicciones, labels=[0, 1]))


if __name__ == "__main__":

    #5. Carga Train y Validation
    ruta = Path(__file__).resolve().parent / "datos_temporales"
    train = pd.read_csv(ruta / "train.csv")
    validation = pd.read_csv(ruta / "validation.csv")

    columnas = [
        "beat_length_ekg", "hr_ekg", "hrv_ekg",
        "dpdt_optical", "hr_optical",
        "invpat_optical", "rpat_optical",
        "arm_angle", "motion_level",
        "steps_last_five_mins", "steps_last_thirty_mins"
    ]

    #6. Define las variables de entrada del modelo
    columnas_delta = []
    for columna in columnas:
        nombre_delta = "delta_" + columna
        columnas_delta.append(nombre_delta)
    columnas = columnas + columnas_delta

    # Prepara datos para XGBoost
    X_train = train[columnas].astype("float32")
    X_val = validation[columnas].astype("float32")
    y_train = train["label"].to_numpy()
    y_val = validation["label"].to_numpy()

    # Verifica que ningún participante de Train esté en Validation
    participantes_train = set(train["pid"])
    participantes_validation = set(validation["pid"])
    participantes_repetidos = participantes_train.intersection(participantes_validation)

    if len(participantes_repetidos) > 0:
        raise ValueError("Hay participantes presentes en Train y Validation.")

    # Comprueba resultados
    print("\nDatos utilizados:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    print("\nParticipantes:")
    print("Train:", train["pid"].nunique())
    print("Validation:", validation["pid"].nunique())

    print("\nDistribucion de mediciones en Train:")
    print(pd.Series(y_train).value_counts().sort_index())

    print("\nValores faltantes en deltas de Train:")
    columnas_delta = [c for c in columnas if c.startswith("delta_")]
    print(X_train[columnas_delta].isna().sum())


    #7. Calcula pesos de clase y prepara los datos para XGBoost
    pesos = calcular_pesos(y_train)

    print("\nPesos de clase:")
    print("Peso clase 0:", pesos[y_train == 0][0])
    print("Peso clase 1:", pesos[y_train == 1][0])
    print("Relacion positivo/negativo:", pesos[y_train == 1][0] / pesos[y_train == 0][0])

    # Prepara DMatrix para XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=pesos)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define parametros del modelo
    parametros = {
        "objective": "binary:logistic",
        "base_score": 0.5,
        "eta": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "device": "cpu",
        "nthread": 1,
        "seed": 2026,
        "disable_default_eval_metric": 1
    }

    #8. Entrenamiento del modelo con el gradiente y Hessiano implementados
    modelo = xgb.train(
        params=parametros,
        dtrain=dtrain,
        num_boost_round=100,
        obj=objetivo_personalizado
    )

    # Obtiene la salida del modelo y la convierte en probabilidades
    margenes = modelo.predict(dval, output_margin=True)
    probabilidades = sigmoide(margenes)

    #9. Evaluacion por participante
    resultados = pd.DataFrame({
        "pid": validation["pid"].to_numpy(),
        "label": y_val,
        "probabilidad": probabilidades
    })

    participantes = resultados.groupby("pid").agg(
        label=("label", "first"),
        probabilidad=("probabilidad", "mean")
    )
    print("\nXGBoost personalizado balanceado")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("\nResultados por participante:")

    # Evaluacion por participante
    evaluar(participantes["label"].to_numpy(),participantes["probabilidad"].to_numpy())


    #10. Valida la funcion objetivo personalizada contra XGBoost oficial
    modelo_oficial = xgb.train(
        params=parametros,
        dtrain=dtrain,
        num_boost_round=100)

    probabilidades_oficial = modelo_oficial.predict(dval)
    diferencia = np.abs(probabilidades - probabilidades_oficial)

    print("\nComparacion con XGBoost oficial:")
    print("Diferencia maxima:", diferencia.max())
    print("Diferencia promedio:", diferencia.mean())
    print(
        "Clasificaciones diferentes:",
        np.sum((probabilidades >= 0.5) !=(probabilidades_oficial >= 0.5))
    )