import pandas as pd
from pathlib import Path

#1. Carga los datos divididos
ruta = Path(__file__).resolve().parent / "datos_divididos"

train = pd.read_csv(ruta / "train.csv")
validation = pd.read_csv(ruta / "validation.csv")
test = pd.read_csv(ruta / "test.csv")

#2. Columnas fisiologicas
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

#3. Calcular variaciones fisiologicas y tiempo entre mediciones consecutivas
def calcular_variaciones(datos):
    datos = datos.copy()
    datos["date_time"] = pd.to_datetime(datos["date_time"],errors="coerce")
    datos = datos.sort_values(["pid", "date_time", "measurement"]).reset_index(drop=True)

    # Tiempo entre mediciones consecutivas
    datos["elapsed_minutes"] = (
        datos.groupby("pid")["date_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    # Diferencias fisiologicas consecutivas
    for columna in columnas:
        datos["delta_" + columna] = (datos.groupby("pid")[columna].diff())

    # Limite provisional de continuidad temporal
    intervalo_maximo = 120
    columnas_delta = []

    for columna in columnas:
        nombre_delta = "delta_" + columna
        columnas_delta.append(nombre_delta)

    # Revisa cada medicion
    for indice in datos.index:
        tiempo = datos.at[indice, "elapsed_minutes"]

        # Si pasaron mas de 120 minutos
        if pd.notna(tiempo) and tiempo > intervalo_maximo:

            # Invalida todas las variaciones de esa medicion
            for columna_delta in columnas_delta:
                datos.at[indice, columna_delta] = float("nan")

    return datos

#4. Analizar intervalos
def analizar_intervalos(nombre, datos):

    intervalos = datos["elapsed_minutes"].dropna()

    print(f"\n{nombre}")
    print("Mediciones:", len(datos))
    print("Participantes:", datos["pid"].nunique())

    print("\nIntervalos en minutos:")
    print(intervalos.describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    ))

    print("Intervalos > 30 min:",(intervalos > 30).sum())
    print("Intervalos > 60 min:",(intervalos > 60).sum())
    print("Intervalos > 120 min:",(intervalos > 120).sum())
    print("Intervalos > 240 min:",(intervalos > 240).sum())


#5. Calcular variaciones
train = calcular_variaciones(train)
validation = calcular_variaciones(validation)
test = calcular_variaciones(test)

print("\nVariaciones calculadas")
print("Train:",train.shape)
print("Validation:",validation.shape)
print("Test:",test.shape)


#6. Analizar intervalos
analizar_intervalos("TRAIN",train)
analizar_intervalos("VALIDATION",validation)
analizar_intervalos("TEST",test)


#7. Ejemplo
print("\nEjemplo de secuencia temporal:")
print(train[["pid","date_time","hr_ekg","delta_hr_ekg","elapsed_minutes"]].head(15))

# Guardar datos temporales
ruta_temporal = (Path(__file__).resolve().parent /"datos_temporales")
ruta_temporal.mkdir(exist_ok=True)

train.to_csv(ruta_temporal / "train.csv",index=False)
validation.to_csv(ruta_temporal / "validation.csv",index=False)
test.to_csv(ruta_temporal / "test.csv",index=False)

print("\nDatos temporales guardados correctamente")