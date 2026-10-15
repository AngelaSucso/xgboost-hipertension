import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 2026

#1. Carga los datos filtrados
ruta = Path(__file__).resolve().parent / "datos_preparados"
datos = pd.read_csv(ruta / "datos_preparados.csv")

# Una fila por participante
participantes = datos[["pid", "label"]].drop_duplicates()

print("Participantes totales:", len(participantes))
print("\nDistribucion total:")
print(participantes["label"].value_counts())


#2. Separa Train (70%) del resto (30%)
train_pid, temporal_pid = train_test_split(
    participantes,
    test_size=0.30, # queda el 30% para validation y test
    stratify=participantes["label"], # mantiene la proporcion de clases
    random_state=SEED
)


#3. Divide el 30% restante en Validation (15%) y Test (15%)
validation_pid, test_pid = train_test_split(
    temporal_pid,
    test_size=0.50, # queda el 15% para test
    stratify=temporal_pid["label"],
    random_state=SEED
)


#4. Asigna todas las mediciones de un mismo participante para cada conjunto
train = datos[datos["pid"].isin(train_pid["pid"])].copy()
validation = datos[datos["pid"].isin(validation_pid["pid"])].copy()
test = datos[datos["pid"].isin(test_pid["pid"])].copy()


#5. Verifica que no haya fuga de participantes
set_train = set(train["pid"])
set_val = set(validation["pid"])
set_test = set(test["pid"])
assert set_train.isdisjoint(set_val)
assert set_train.isdisjoint(set_test)
assert set_val.isdisjoint(set_test)


#6. Comprueba resultados
def mostrar(nombre, df):
    participantes_df = df[["pid", "label"]].drop_duplicates()

    print(f"\n{nombre}")
    print("Participantes:", len(participantes_df))
    print("Mediciones:", len(df))

    print("Distribucion:")
    print(participantes_df["label"].value_counts().sort_index())

mostrar("TRAIN", train)
mostrar("VALIDATION", validation)
mostrar("TEST", test)


#7. Guarda
ruta_salida = (Path(__file__).resolve().parent /"datos_divididos")
ruta_salida.mkdir(exist_ok=True)

train.to_csv(ruta_salida / "train.csv",index=False)
validation.to_csv(ruta_salida / "validation.csv",index=False)
test.to_csv(ruta_salida / "test.csv",index=False)

print("\nArchivos guardados en:", ruta_salida)