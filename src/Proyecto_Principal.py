# =============================================================================
# PROYECTO FINAL - APLICACIONES BIG DATA I
# Metodología: CRISP-DM
# Dataset: Student Productivity & Distraction Dataset (20.000 registros)
# Asignatura: Aplicaciones de Big Data I - 4º Título Superior en Data Science
# ESIC University - Curso 2025/2026
# =============================================================================
#
# DESCRIPCIÓN DEL PROYECTO:
# Este proyecto analiza cómo las distracciones digitales (móvil, redes sociales,
# YouTube, gaming) afectan a la productividad y rendimiento académico de los
# estudiantes. Se entrena un modelo de clasificación que predice si un estudiante
# tendrá productividad ALTA, MEDIA o BAJA en función de sus hábitos.
#
# El objetivo final es desplegar este modelo como una API REST que permita
# consultar predicciones y estadísticas del dataset.
# =============================================================================

# --- Importación de librerías necesarias ---
# pandas: para manejar el dataset como DataFrame (tablas de datos)
# numpy: para operaciones numéricas y arrays
# matplotlib: para generar gráficos
# sqlite3: para la persistencia de datos en base de datos relacional
# sklearn: para entrenar y evaluar modelos de Machine Learning
# pickle: para guardar el modelo entrenado en un fichero

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sqlite3
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =============================================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# =============================================================================
# En esta fase definimos el problema desde el punto de vista del negocio.
# Pregunta clave: ¿Qué factores de distracción digital afectan más a la
# productividad de los estudiantes universitarios?
#
# Objetivo de negocio:
# - Identificar patrones de comportamiento que predicen baja productividad
# - Proporcionar a través de una API predicciones y estadísticas que permitan
#   a orientadores académicos y profesores tomar decisiones informadas
#
# Objetivo de minería de datos:
# - Clasificar a los estudiantes en tres niveles de productividad:
#   BAJA (0-33), MEDIA (34-66), ALTA (67-100)
# - Identificar qué variables tienen más influencia en la productividad
# =============================================================================

print("=" * 65)
print("PROYECTO FINAL - CRISP-DM")
print("Student Productivity & Distraction Analysis")
print("=" * 65)
print()
print("FASE 1: COMPRENSIÓN DEL NEGOCIO")
print("-" * 40)
print("Objetivo: Predecir la productividad del estudiante (ALTA/MEDIA/BAJA)")
print("basándose en sus hábitos de estudio y uso de tecnología.")
print()

# =============================================================================
# FASE 2: COMPRENSIÓN DE LOS DATOS
# =============================================================================
# En esta fase cargamos el dataset y lo exploramos para entender qué
# información contiene, cuántos registros tiene, qué tipos de variables hay,
# y si existen valores nulos o anomalías evidentes.
# =============================================================================

print("FASE 2: COMPRENSIÓN DE LOS DATOS")
print("-" * 40)

# --- Cargamos el dataset desde el fichero CSV ---
RUTA_DATASET = "student_productivity_distraction_dataset_20000.csv"
df = pd.read_csv(RUTA_DATASET)

print(f"Dataset cargado correctamente.")
print(f"Número de filas (registros): {df.shape[0]}")
print(f"Número de columnas (variables): {df.shape[1]}")
print()

# --- Mostramos las primeras filas para ver el aspecto del dataset ---
print("Primeras 5 filas del dataset:")
print(df.head())
print()

# --- Información general: tipos de datos y valores nulos ---
print("Información del dataset (tipos de datos):")
print(df.dtypes)
print()

# --- Catálogo de datos: descripción de cada variable ---
# Esta información es importante para la fase de comprensión del negocio
catalogo = {
    "student_id":               "Identificador único del estudiante (numérica discreta)",
    "age":                      "Edad del estudiante en años (numérica continua)",
    "gender":                   "Género: Male / Female (categórica nominal)",
    "study_hours_per_day":      "Horas diarias dedicadas al estudio (numérica continua)",
    "sleep_hours":              "Horas de sueño por noche (numérica continua)",
    "phone_usage_hours":        "Horas diarias de uso del teléfono (numérica continua)",
    "social_media_hours":       "Horas diarias en redes sociales (numérica continua)",
    "youtube_hours":            "Horas diarias viendo YouTube (numérica continua)",
    "gaming_hours":             "Horas diarias jugando videojuegos (numérica continua)",
    "breaks_per_day":           "Número de descansos al día (numérica discreta)",
    "coffee_intake_mg":         "Consumo diario de cafeína en mg (numérica continua)",
    "exercise_minutes":         "Minutos de ejercicio al día (numérica continua)",
    "assignments_completed":    "Número de tareas completadas (numérica discreta)",
    "attendance_percentage":    "Porcentaje de asistencia a clase (numérica continua)",
    "stress_level":             "Nivel de estrés del 1 al 10 (numérica discreta / ordinal)",
    "focus_score":              "Puntuación de concentración del 1 al 100 (numérica continua)",
    "final_grade":              "Nota final del curso (numérica continua)",
    "productivity_score":       "Puntuación de productividad del 0 al 100 (VARIABLE OBJETIVO)"
}

print("CATÁLOGO DE DATOS:")
for variable, descripcion in catalogo.items():
    print(f"  - {variable}: {descripcion}")
print()

# --- Estadísticos descriptivos básicos ---
print("Estadísticos descriptivos:")
print(df.describe().round(2))
print()

# --- Comprobación de valores nulos ---
nulos = df.isnull().sum()
print("Valores nulos por columna:")
print(nulos)
print()

# --- Variables categóricas: cuántas categorías distintas tiene cada una ---
# Identificamos las columnas de tipo object (texto), que son las categóricas
columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
print(f"Variables categóricas encontradas: {columnas_categoricas}")
for col in columnas_categoricas:
    print(f"  Valores únicos en '{col}': {df[col].unique()}")
print()

# =============================================================================
# FASE 3: PREPARACIÓN DE LOS DATOS (Data Preparation)
# =============================================================================
# En esta fase limpiamos y transformamos los datos para que puedan ser
# usados por el modelo de Machine Learning.
#
# Tareas que realizamos:
# 1. Eliminar columnas innecesarias (student_id no aporta información)
# 2. Gestionar valores nulos (si los hubiera)
# 3. Crear la variable objetivo categórica (productividad_nivel)
# 4. Codificar las variables categóricas en números
# 5. Crear nuevas variables derivadas (feature engineering básico)
# =============================================================================

print("FASE 3: PREPARACIÓN DE LOS DATOS")
print("-" * 40)

# --- Paso 1: Eliminar columnas que no aportan valor al modelo ---
# student_id es solo un identificador, no tiene relación con la productividad
df = df.drop(columns=["student_id"])
print("Columna 'student_id' eliminada (no aporta información al modelo).")

# --- Paso 2: Gestión de valores nulos ---
# Como comprobamos anteriormente, no hay nulos, pero lo verificamos de nuevo
if df.isnull().sum().sum() == 0:
    print("No se encontraron valores nulos. El dataset está completo.")
else:
    # Si hubiera nulos en columnas numéricas, los rellenamos con la media
    # Si hubiera nulos en columnas categóricas, los rellenamos con la moda
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    print(f"Valores nulos rellenados.")
print()

# --- Paso 3: Crear la variable objetivo categórica ---
# productivity_score es un número entre 0 y 100.
# Lo convertimos en 3 categorías:
#   - BAJA:  productivity_score < 34
#   - MEDIA: productivity_score entre 34 y 66
#   - ALTA:  productivity_score > 66
#
# Usamos pd.cut() que divide un rango numérico en intervalos (bins)

def categorizar_productividad(score):
    """
    Función que recibe un valor numérico de productividad (0-100)
    y devuelve una etiqueta categórica: 'BAJA', 'MEDIA' o 'ALTA'.
    Esta función se aplica a cada fila del DataFrame.
    """
    if score < 34:
        return "BAJA"
    elif score <= 66:
        return "MEDIA"
    else:
        return "ALTA"

# Aplicamos la función a cada valor de la columna productivity_score
# apply() ejecuta la función en cada fila
df["productividad_nivel"] = df["productivity_score"].apply(categorizar_productividad)

print("Variable objetivo creada: 'productividad_nivel'")
print("Distribución de clases:")
print(df["productividad_nivel"].value_counts())
print()

# --- Paso 4: Codificación de variables categóricas ---
# Los modelos de Machine Learning solo trabajan con números.
# LabelEncoder transforma texto en números: Female=0, Male=1

# Hacemos una copia del DataFrame original para no perder los datos originales
df_modelo = df.copy()

# Codificamos la variable 'gender'
encoder_gender = LabelEncoder()
df_modelo["gender"] = encoder_gender.fit_transform(df_modelo["gender"])
print(f"Variable 'gender' codificada: {list(encoder_gender.classes_)} → {list(range(len(encoder_gender.classes_)))}")

# Codificamos la variable objetivo 'productividad_nivel'
encoder_productividad = LabelEncoder()
df_modelo["productividad_nivel"] = encoder_productividad.fit_transform(df_modelo["productividad_nivel"])
print(f"Variable 'productividad_nivel' codificada: {list(encoder_productividad.classes_)} → {list(range(len(encoder_productividad.classes_)))}")
print()

# --- Paso 5: Feature Engineering ---
# Creamos nuevas variables que pueden ser útiles para el modelo:
# - total_distraccion: suma de horas en actividades distractoras
# - ratio_estudio_distraccion: comparación entre horas de estudio y distracción

df_modelo["total_distraccion"] = (
    df_modelo["phone_usage_hours"] +
    df_modelo["social_media_hours"] +
    df_modelo["youtube_hours"] +
    df_modelo["gaming_hours"]
)

# Evitamos división por cero con un pequeño valor (0.001)
df_modelo["ratio_estudio_distraccion"] = (
    df_modelo["study_hours_per_day"] /
    (df_modelo["total_distraccion"] + 0.001)
)

print("Variables derivadas creadas:")
print("  - 'total_distraccion': suma total de horas en actividades distractoras")
print("  - 'ratio_estudio_distraccion': ratio horas estudio / horas distracción")
print()

# =============================================================================
# FASE 4: MODELADO (Modeling)
# =============================================================================
# Entrenamos un modelo de clasificación con Decision Tree (Árbol de Decisión).
# Este algoritmo es fácil de interpretar y adecuado para clasificación multiclase.
#
# Pasos:
# 1. Separar características (X) de la variable objetivo (y)
# 2. Dividir en conjunto de entrenamiento y de prueba (80%/20%)
# 3. Entrenar el modelo
# 4. Guardar el modelo entrenado en un fichero .pkl
# =============================================================================

print("FASE 4: MODELADO")
print("-" * 40)

# --- Paso 1: Separar características (X) y variable objetivo (y) ---
# X: todas las columnas EXCEPTO productivity_score (numérico) y
#    productividad_nivel (categórico, es lo que queremos predecir)
# y: únicamente la variable que queremos predecir

columnas_excluir = ["productivity_score", "productividad_nivel"]
X = df_modelo.drop(columns=columnas_excluir)
y = df_modelo["productividad_nivel"]

print(f"Variables de entrada (X): {list(X.columns)}")
print(f"Variable objetivo (y): productividad_nivel")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")
print()

# --- Paso 2: Dividir en entrenamiento y prueba ---
# train_test_split divide aleatoriamente el dataset:
# - 80% para entrenar el modelo
# - 20% para evaluar su rendimiento con datos que no ha visto
# random_state=42 asegura que la división sea siempre la misma (reproducible)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para test
    random_state=42,    # Semilla para reproducibilidad
    stratify=y          # Mantiene la misma proporción de clases en train y test
)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} filas")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} filas")
print()

# --- Paso 3: Entrenar el modelo Decision Tree ---
# max_depth=5 limita la profundidad del árbol para evitar overfitting
# (sobreajuste: que el modelo funcione bien en train pero mal en datos nuevos)

modelo = DecisionTreeClassifier(
    max_depth=5,        # Profundidad máxima del árbol
    random_state=42     # Reproducibilidad
)

# fit() es el método que ENTRENA el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)
print("Modelo Decision Tree entrenado correctamente.")
print()

# =============================================================================
# FASE 5: EVALUACIÓN (Evaluation)
# =============================================================================
# Comprobamos qué tan bien funciona el modelo con datos que no ha visto
# durante el entrenamiento (los datos de test).
#
# Métricas que usamos:
# - Accuracy: porcentaje de predicciones correctas
# - Classification Report: precisión, recall y f1-score por clase
# - Matriz de confusión: tabla que muestra errores del modelo
# =============================================================================

print("FASE 5: EVALUACIÓN DEL MODELO")
print("-" * 40)

# --- Realizamos predicciones con el conjunto de prueba ---
# predict() aplica el modelo entrenado a datos nuevos (X_test)
y_predicho = modelo.predict(X_test)

# --- Accuracy: porcentaje de predicciones correctas ---
accuracy = accuracy_score(y_test, y_predicho)
print(f"Accuracy (precisión global) del modelo: {accuracy * 100:.2f}%")
print()

# --- Informe de clasificación completo ---
# Muestra precisión, recall y f1-score para cada clase (ALTA, BAJA, MEDIA)
nombres_clases = encoder_productividad.classes_
print("Informe de clasificación detallado:")
print(classification_report(y_test, y_predicho, target_names=nombres_clases))

# --- Importancia de las variables ---
# feature_importances_ indica qué columnas son más importantes para el modelo
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)
print("Variables más importantes según el modelo:")
print(importancias.head(10).round(4))
print()

# =============================================================================
# FASE 6: GUARDAR ARTEFACTOS DEL MODELO
# =============================================================================
# Guardamos el modelo y los encoders en ficheros .pkl para poder usarlos
# desde la API sin necesidad de reentrenar cada vez.
# pickle.dump() serializa (convierte a bytes) el objeto Python y lo guarda.
# =============================================================================

print("GUARDANDO ARTEFACTOS DEL MODELO")
print("-" * 40)

# Guardamos el modelo entrenado
with open("modelos/modelo_productividad.pkl", "wb") as f:
    pickle.dump(modelo, f)
print("Modelo guardado en: modelos/modelo_productividad.pkl")

# Guardamos el encoder de género
with open("modelos/encoder_gender.pkl", "wb") as f:
    pickle.dump(encoder_gender, f)
print("Encoder de género guardado en: modelos/encoder_gender.pkl")

# Guardamos el encoder de productividad
with open("modelos/encoder_productividad.pkl", "wb") as f:
    pickle.dump(encoder_productividad, f)
print("Encoder de productividad guardado en: modelos/encoder_productividad.pkl")

# Guardamos los nombres de las columnas de entrada del modelo
columnas_modelo = list(X.columns)
with open("modelos/columnas_modelo.pkl", "wb") as f:
    pickle.dump(columnas_modelo, f)
print(f"Columnas del modelo guardadas: {columnas_modelo}")
print()

# =============================================================================
# GENERACIÓN DE GRÁFICOS (para la documentación y comprensión de datos)
# =============================================================================
# Generamos varios gráficos que ayudan a entender el dataset y los resultados.
# Usamos matplotlib como se ve en los apuntes de clase.
# =============================================================================

print("GENERANDO GRÁFICOS")
print("-" * 40)

# --- Gráfico 1: Distribución de la variable objetivo ---
# Vemos cuántos estudiantes hay en cada categoría de productividad
plt.figure(figsize=(8, 5))
conteo_clases = df["productividad_nivel"].value_counts()
# Definimos colores para cada barra
colores = ['#e74c3c', '#f39c12', '#2ecc71']
barras = plt.bar(conteo_clases.index, conteo_clases.values, color=colores)
plt.title("Distribución de Niveles de Productividad", fontsize=14)
plt.xlabel("Nivel de Productividad")
plt.ylabel("Número de Estudiantes")
# Añadimos el valor encima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width() / 2., altura + 50,
             f'{int(altura)}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("graficos/01_distribucion_productividad.png", dpi=100)
plt.close()
print("Gráfico 1 guardado: distribucion_productividad.png")

# --- Gráfico 2: Horas de estudio vs Productividad ---
plt.figure(figsize=(8, 5))
# Agrupamos por nivel de productividad y calculamos la media de horas de estudio
media_estudio = df.groupby("productividad_nivel")["study_hours_per_day"].mean()
# Ordenamos las categorías lógicamente
orden = ["BAJA", "MEDIA", "ALTA"]
media_estudio = media_estudio.reindex(orden)
plt.bar(media_estudio.index, media_estudio.values, color=['#e74c3c', '#f39c12', '#2ecc71'])
plt.title("Media de Horas de Estudio por Nivel de Productividad", fontsize=13)
plt.xlabel("Nivel de Productividad")
plt.ylabel("Media Horas de Estudio/Día")
plt.tight_layout()
plt.savefig("graficos/02_estudio_vs_productividad.png", dpi=100)
plt.close()
print("Gráfico 2 guardado: estudio_vs_productividad.png")

# --- Gráfico 3: Importancia de variables del modelo ---
plt.figure(figsize=(10, 6))
# Mostramos las 10 variables más importantes
importancias_top10 = importancias.head(10)
plt.barh(importancias_top10.index[::-1], importancias_top10.values[::-1], color='steelblue')
plt.title("Top 10 Variables más Importantes en el Modelo", fontsize=13)
plt.xlabel("Importancia (Gini)")
plt.tight_layout()
plt.savefig("graficos/03_importancia_variables.png", dpi=100)
plt.close()
print("Gráfico 3 guardado: importancia_variables.png")

# --- Gráfico 4: Histograma de Horas de Distracción Total ---
plt.figure(figsize=(8, 5))
df["total_distraccion_calc"] = (
    df["phone_usage_hours"] + df["social_media_hours"] +
    df["youtube_hours"] + df["gaming_hours"]
)
plt.hist(df["total_distraccion_calc"], bins=30, color='coral', edgecolor='black')
plt.title("Distribución de Horas Totales de Distracción", fontsize=13)
plt.xlabel("Horas Totales de Distracción al Día")
plt.ylabel("Número de Estudiantes")
plt.tight_layout()
plt.savefig("graficos/04_distribucion_distraccion.png", dpi=100)
plt.close()
print("Gráfico 4 guardado: distribucion_distraccion.png")

# --- Gráfico 5: Relación entre Nota Final y Productividad ---
plt.figure(figsize=(8, 5))
media_nota = df.groupby("productividad_nivel")["final_grade"].mean()
media_nota = media_nota.reindex(orden)
plt.bar(media_nota.index, media_nota.values, color=['#e74c3c', '#f39c12', '#2ecc71'])
plt.title("Media de la Nota Final por Nivel de Productividad", fontsize=13)
plt.xlabel("Nivel de Productividad")
plt.ylabel("Media Nota Final")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("graficos/05_nota_vs_productividad.png", dpi=100)
plt.close()
print("Gráfico 5 guardado: nota_vs_productividad.png")
print()

# =============================================================================
# PERSISTENCIA DE DATOS - BASE DE DATOS SQLITE
# =============================================================================
# Guardamos el dataset procesado en una base de datos SQLite.
# SQLite es una base de datos relacional ligera que no requiere servidor.
# Usamos sqlite3, el módulo estándar de Python para trabajar con SQLite.
#
# Creamos las siguientes tablas:
# - estudiantes: todos los datos del dataset
# - predicciones_api: registro de predicciones realizadas por la API
# =============================================================================

print("CREANDO BASE DE DATOS SQLITE")
print("-" * 40)

# Nos conectamos a la base de datos (si no existe, la crea)
conexion = sqlite3.connect("data/estudiantes.db")
cursor = conexion.cursor()

# --- Crear tabla 'estudiantes' ---
# DROP TABLE IF EXISTS evita errores si la tabla ya existía de una ejecución anterior
cursor.execute("DROP TABLE IF EXISTS estudiantes")

# CREATE TABLE define la estructura de la tabla con sus columnas y tipos
cursor.execute("""
    CREATE TABLE estudiantes (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        age             INTEGER,
        gender          TEXT,
        study_hours     REAL,
        sleep_hours     REAL,
        phone_usage     REAL,
        social_media    REAL,
        youtube_hours   REAL,
        gaming_hours    REAL,
        breaks_per_day  INTEGER,
        coffee_intake   REAL,
        exercise_min    REAL,
        assignments     INTEGER,
        attendance_pct  REAL,
        stress_level    INTEGER,
        focus_score     REAL,
        final_grade     REAL,
        productivity    REAL,
        nivel_productividad TEXT
    )
""")

# --- Insertar los datos del dataset en la tabla ---
# Construimos una lista de tuplas con los datos de cada fila
datos_insertar = []
for _, fila in df.iterrows():
    datos_insertar.append((
        int(fila["age"]),
        fila["gender"],
        float(fila["study_hours_per_day"]),
        float(fila["sleep_hours"]),
        float(fila["phone_usage_hours"]),
        float(fila["social_media_hours"]),
        float(fila["youtube_hours"]),
        float(fila["gaming_hours"]),
        int(fila["breaks_per_day"]),
        float(fila["coffee_intake_mg"]),
        float(fila["exercise_minutes"]),
        int(fila["assignments_completed"]),
        float(fila["attendance_percentage"]),
        int(fila["stress_level"]),
        float(fila["focus_score"]),
        float(fila["final_grade"]),
        float(fila["productivity_score"]),
        fila["productividad_nivel"]
    ))

# executemany() inserta múltiples filas de una sola vez (más eficiente)
cursor.executemany("""
    INSERT INTO estudiantes
    (age, gender, study_hours, sleep_hours, phone_usage, social_media,
     youtube_hours, gaming_hours, breaks_per_day, coffee_intake,
     exercise_min, assignments, attendance_pct, stress_level,
     focus_score, final_grade, productivity, nivel_productividad)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
""", datos_insertar)

# --- Crear tabla para registrar predicciones de la API ---
cursor.execute("DROP TABLE IF EXISTS predicciones_api")
cursor.execute("""
    CREATE TABLE predicciones_api (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha           TEXT,
        study_hours     REAL,
        sleep_hours     REAL,
        phone_usage     REAL,
        social_media    REAL,
        youtube_hours   REAL,
        gaming_hours    REAL,
        prediccion      TEXT,
        probabilidad    REAL
    )
""")

# Guardamos los cambios en la base de datos
conexion.commit()

# Verificamos que los datos se insertaron correctamente
cursor.execute("SELECT COUNT(*) FROM estudiantes")
total_registros = cursor.fetchone()[0]
print(f"Base de datos creada: data/estudiantes.db")
print(f"Registros insertados en tabla 'estudiantes': {total_registros}")

# Cerramos la conexión
conexion.close()
print()

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("=" * 65)
print("RESUMEN DEL PROYECTO")
print("=" * 65)
print(f"  Dataset: {total_registros} estudiantes analizados")
print(f"  Variables: {len(catalogo)} columnas en el dataset original")
print(f"  Modelo: Decision Tree Classifier")
print(f"  Accuracy obtenida: {accuracy * 100:.2f}%")
print(f"  Clases predichas: {list(encoder_productividad.classes_)}")
print(f"  Artefactos guardados en: modelos/")
print(f"  Gráficos guardados en: graficos/")
print(f"  Base de datos: data/estudiantes.db")
print()
print("El modelo está listo para ser desplegado como API.")
print("Ejecuta 'python api.py' para iniciar el servidor FastAPI.")
print("=" * 65)
