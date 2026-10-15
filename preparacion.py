import numpy as np
import pandas as pd
from pathlib import Path

#Ubicacion del dataset AURORA
ruta_aurora = Path("C:/Users/Angela/Desktop/AURORA")

#Carga archivos
participantes = pd.read_csv(ruta_aurora / "participants.tsv",sep="\t")
mediciones = pd.read_csv(ruta_aurora / "measurements_oscillometric.tsv",sep="\t")
caracteristicas = pd.read_csv(ruta_aurora / "features.tsv",sep="\t")


#1. Selecciona participantes y definir etiquetas
participantes = participantes[participantes["self_report_htn"].isin(["norm", "unmanaged"])].copy()
participantes["label"] = participantes["self_report_htn"].map({"norm": 0,"unmanaged": 1})


#2. Selecciona fase ambulatoria
mediciones = mediciones[mediciones["phase"] == "ambulatory"].copy()


#3. Caracteristicas fisiologicas
columnas = [
    "beat_length_ekg",
    "hr_ekg",
    "hrv_ekg",
    "dpdt_optical",
    "hr_optical",
    "invpat_optical",
    "rpat_optical",
    "arm_angle",
    "motion_level",
    "steps_last_five_mins",
    "steps_last_thirty_mins"
]

# Define grupos de características
ecg = ["beat_length_ekg","hr_ekg","hrv_ekg"]
ppg = ["dpdt_optical","hr_optical","invpat_optical","rpat_optical"]

# Define claves para unir tablas
#   id: Identificador del participante
#   phase: Fase de medición (ambulatory)
#   measurement: Identificador de la medición
claves = ["pid","phase","measurement"]


#4. Obtiene features de features.tsv y las une con mediciones
datos_features = caracteristicas[claves + ["date_time", "quality_optical"] + columnas].copy()

#Une measurements con features
datos = pd.merge(
    mediciones[claves + ["date_time"]],
    datos_features,
    on=claves,
    how="inner",
    suffixes=("_medicion", "_feature"),
    validate="one_to_one"
)

#Agrega etiqueta del participante
datos = pd.merge(
    datos,
    participantes[["pid", "label"]],
    on="pid",
    how="inner",
    validate="many_to_one"
)


#5. Limpia valores
columnas_numericas = columnas + ["quality_optical"]

# Convierte a numerico y reemplaza inf por NaN
datos[columnas_numericas] = datos[columnas_numericas].apply(pd.to_numeric,errors="coerce")
datos[columnas_numericas] = datos[columnas_numericas].replace([np.inf, -np.inf],np.nan)
# Necesario para ordenar cronologicamente las mediciones
datos["date_time"] = pd.to_datetime(datos["date_time_medicion"],errors="coerce")


#6. Selecciona mediciones fisiologicas utiles
datos = datos[
    datos["date_time"].notna() # fecha valida
    & datos[ecg].notna().all(axis=1) # todas las características ECG presentes
    & datos[ppg].notna().any(axis=1) # al menos una característica PPG presente
    & datos["quality_optical"].notna()].copy()
print("Mediciones antes del filtro de calidad:",len(datos))

# Control de calidad PPG
datos = datos[datos["quality_optical"] >= 0.65].copy()
print("Mediciones despues del filtro de calidad:",len(datos))


#7. Selecciona participantes con suficiente info temporal
conteo = datos.groupby("pid").size()
# Participantes con al menos 20 mediciones
participantes_validos = conteo[conteo >= 20].index
# Elimina participantes con menos de 20 mediciones
datos = datos[datos["pid"].isin(participantes_validos)].copy()

# Orden cronologico
datos = datos.sort_values(["pid", "date_time"]).reset_index(drop=True)


#8. Comprueba resultados
print("\nParticipantes finales:", datos["pid"].nunique())
print("Mediciones finales:", len(datos))

print("\nDistribucion por clase:")
print(datos.groupby("label")["pid"].nunique())

print("\nMediciones por participante:")
print(datos.groupby("pid").size().describe())


#9. Guarda datos preparados
ruta_salida = Path(__file__).resolve().parent / "datos_preparados"
ruta_salida.mkdir(exist_ok=True)

columnas_salida = [
    "pid",
    "phase",
    "measurement",
    "date_time",
    "label",
    "quality_optical"
] + columnas

datos[columnas_salida].to_csv(ruta_salida / "datos_preparados.csv",index=False)
print("\nArchivo guardado en:", ruta_salida / "datos_preparados.csv")