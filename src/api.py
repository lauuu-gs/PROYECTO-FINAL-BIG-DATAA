# =============================================================================
# API REST - STUDENT PRODUCTIVITY PREDICTION
# Framework: FastAPI
# Fase CRISP-DM: Despliegue (Deployment)
# =============================================================================
#
# Esta API permite:
#   GET  /                           → Bienvenida y listado de endpoints
#   GET  /estadisticas               → Estadísticas generales del dataset
#   GET  /estadisticas/genero        → Estadísticas agrupadas por género
#   GET  /estadisticas/productividad → Estadísticas por nivel de productividad
#   GET  /estudiantes                → Lista de estudiantes (paginada)
#   GET  /estudiante/{id}            → Datos de un estudiante específico
#   POST /predecir                   → Predice el nivel de productividad
#   GET  /predicciones               → Historial de predicciones realizadas
#   PUT  /estudiante/{id}            → Actualiza datos de un estudiante
#   DELETE /estudiante/{id}          → Elimina un estudiante
#   POST /consejo                    → Genera consejos personalizados con IA (Groq)
#
# Para arrancar el servidor:
#   python api.py
#
# Documentación interactiva (Swagger):
#   http://localhost:5000/docs
# =============================================================================
 
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import sqlite3
import pickle
import numpy as np
from datetime import datetime
import uvicorn
import os
import requests as http_requests
 
# =============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN FASTAPI
# =============================================================================
 
app = FastAPI(
    title="Student Productivity Prediction API",
    description="API para análisis y predicción de productividad estudiantil basada en CRISP-DM",
    version="1.0"
)
 
# --- Rutas a los ficheros del modelo y la base de datos ---
RUTA_MODELO    = "modelos/modelo_productividad.pkl"
RUTA_ENCODER_G = "modelos/encoder_gender.pkl"
RUTA_ENCODER_P = "modelos/encoder_productividad.pkl"
RUTA_COLUMNAS  = "modelos/columnas_modelo.pkl"
RUTA_BD        = "data/estudiantes.db"
 
# =============================================================================
# CARGA DE ARTEFACTOS DEL MODELO
# =============================================================================
 
def cargar_modelo():
    try:
        with open(RUTA_MODELO, "rb") as f:
            modelo = pickle.load(f)
        with open(RUTA_ENCODER_G, "rb") as f:
            enc_gender = pickle.load(f)
        with open(RUTA_ENCODER_P, "rb") as f:
            enc_productividad = pickle.load(f)
        with open(RUTA_COLUMNAS, "rb") as f:
            columnas = pickle.load(f)
        print("Modelo y encoders cargados correctamente.")
        return modelo, enc_gender, enc_productividad, columnas
    except Exception as error:
        print(f"ERROR al cargar el modelo: {error}")
        return None, None, None, None
 
modelo, enc_gender, enc_productividad, columnas_modelo = cargar_modelo()
 
# =============================================================================
# FUNCIÓN AUXILIAR: CONEXIÓN A LA BASE DE DATOS
# =============================================================================
 
def obtener_conexion():
    conexion = sqlite3.connect(RUTA_BD)
    conexion.row_factory = sqlite3.Row
    return conexion
 
# =============================================================================
# MODELOS PYDANTIC (esquemas de entrada para FastAPI)
# =============================================================================
 
class DatosEstudiante(BaseModel):
    age: int
    gender: str
    study_hours_per_day: float
    sleep_hours: float
    phone_usage_hours: float
    social_media_hours: float
    youtube_hours: float
    gaming_hours: float
    breaks_per_day: int
    coffee_intake_mg: float
    exercise_minutes: float
    assignments_completed: int
    attendance_percentage: float
    stress_level: int
    focus_score: float
    final_grade: float
 
    class Config:
        json_schema_extra = {
            "example": {
                "age": 22,
                "gender": "Female",
                "study_hours_per_day": 5.0,
                "sleep_hours": 7.0,
                "phone_usage_hours": 2.0,
                "social_media_hours": 1.0,
                "youtube_hours": 1.0,
                "gaming_hours": 0.5,
                "breaks_per_day": 3,
                "coffee_intake_mg": 200,
                "exercise_minutes": 30,
                "assignments_completed": 10,
                "attendance_percentage": 85.0,
                "stress_level": 4,
                "focus_score": 70,
                "final_grade": 75.0
            }
        }
 
class DatosActualizacion(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    study_hours: Optional[float] = None
    sleep_hours: Optional[float] = None
    phone_usage: Optional[float] = None
    social_media: Optional[float] = None
    youtube_hours: Optional[float] = None
    gaming_hours: Optional[float] = None
    breaks_per_day: Optional[int] = None
    coffee_intake: Optional[float] = None
    exercise_min: Optional[float] = None
    assignments: Optional[int] = None
    attendance_pct: Optional[float] = None
    stress_level: Optional[int] = None
    focus_score: Optional[float] = None
    final_grade: Optional[float] = None
 
# =============================================================================
# ENDPOINT 1: PÁGINA DE BIENVENIDA
# GET /
# =============================================================================
 
@app.get("/", tags=["Info"])
def inicio():
    """Endpoint raíz: devuelve información sobre la API y los endpoints disponibles."""
    return {
        "api": "Student Productivity Prediction API",
        "version": "1.0",
        "descripcion": "API para análisis de productividad estudiantil basada en CRISP-DM",
        "documentacion_swagger": "http://localhost:5000/docs",
        "endpoints": {
            "GET  /":                           "Esta ayuda",
            "GET  /estadisticas":               "Estadísticas generales del dataset",
            "GET  /estadisticas/genero":        "Estadísticas por género",
            "GET  /estadisticas/productividad": "Estadísticas por nivel de productividad",
            "GET  /estudiantes":                "Lista de estudiantes (param: pagina, limite, nivel)",
            "GET  /estudiante/{id}":            "Datos de un estudiante por ID",
            "POST /predecir":                   "Predice nivel de productividad",
            "GET  /predicciones":               "Historial de predicciones realizadas",
            "PUT  /estudiante/{id}":            "Actualiza datos de un estudiante",
            "DELETE /estudiante/{id}":          "Elimina un estudiante"
        }
    }
 
# =============================================================================
# ENDPOINT 2: ESTADÍSTICAS GENERALES
# GET /estadisticas
# =============================================================================
 
@app.get("/estadisticas", tags=["Estadísticas"])
def estadisticas_generales():
    """Devuelve estadísticas descriptivas generales del dataset."""
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        cursor.execute("""
            SELECT
                COUNT(*) AS total_estudiantes,
                ROUND(AVG(study_hours), 2) AS media_horas_estudio,
                ROUND(AVG(sleep_hours), 2) AS media_horas_sueno,
                ROUND(AVG(phone_usage), 2) AS media_uso_movil,
                ROUND(AVG(social_media), 2) AS media_redes_sociales,
                ROUND(AVG(gaming_hours), 2) AS media_gaming,
                ROUND(AVG(final_grade), 2) AS media_nota_final,
                ROUND(AVG(productivity), 2) AS media_productividad,
                ROUND(AVG(stress_level), 2) AS media_estres,
                ROUND(AVG(focus_score), 2) AS media_concentracion
            FROM estudiantes
        """)
        stats = dict(cursor.fetchone())
 
        cursor.execute("""
            SELECT nivel_productividad, COUNT(*) AS total
            FROM estudiantes
            GROUP BY nivel_productividad
            ORDER BY nivel_productividad
        """)
        dist_productividad = {fila["nivel_productividad"]: fila["total"] for fila in cursor.fetchall()}
 
        cursor.execute("""
            SELECT gender, COUNT(*) AS total
            FROM estudiantes
            GROUP BY gender
            ORDER BY gender
        """)
        dist_genero = {fila["gender"]: fila["total"] for fila in cursor.fetchall()}
 
        conexion.close()
 
        return {
            "estadisticas_generales": stats,
            "distribucion_productividad": dist_productividad,
            "distribucion_genero": dist_genero
        }
 
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 3: ESTADÍSTICAS POR GÉNERO
# GET /estadisticas/genero
# =============================================================================
 
@app.get("/estadisticas/genero", tags=["Estadísticas"])
def estadisticas_por_genero():
    """Devuelve estadísticas agrupadas por género."""
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        cursor.execute("""
            SELECT
                gender,
                COUNT(*) AS total,
                ROUND(AVG(study_hours), 2) AS media_estudio,
                ROUND(AVG(phone_usage + social_media + youtube_hours + gaming_hours), 2)
                    AS media_distraccion_total,
                ROUND(AVG(final_grade), 2) AS media_nota,
                ROUND(AVG(productivity), 2) AS media_productividad,
                ROUND(AVG(stress_level), 2) AS media_estres
            FROM estudiantes
            GROUP BY gender
            ORDER BY gender
        """)
 
        resultados = [dict(fila) for fila in cursor.fetchall()]
        conexion.close()
 
        return {"estadisticas_por_genero": resultados}
 
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 4: ESTADÍSTICAS POR NIVEL DE PRODUCTIVIDAD
# GET /estadisticas/productividad
# =============================================================================
 
@app.get("/estadisticas/productividad", tags=["Estadísticas"])
def estadisticas_por_productividad():
    """Devuelve estadísticas agrupadas por nivel de productividad (ALTA/MEDIA/BAJA)."""
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        cursor.execute("""
            SELECT
                nivel_productividad,
                COUNT(*) AS total_estudiantes,
                ROUND(AVG(study_hours), 2) AS media_estudio,
                ROUND(AVG(sleep_hours), 2) AS media_sueno,
                ROUND(AVG(phone_usage), 2) AS media_movil,
                ROUND(AVG(social_media), 2) AS media_redes,
                ROUND(AVG(gaming_hours), 2) AS media_gaming,
                ROUND(AVG(final_grade), 2) AS media_nota,
                ROUND(AVG(stress_level), 2) AS media_estres,
                ROUND(AVG(focus_score), 2) AS media_concentracion
            FROM estudiantes
            GROUP BY nivel_productividad
            ORDER BY CASE nivel_productividad
                WHEN 'ALTA' THEN 1
                WHEN 'MEDIA' THEN 2
                WHEN 'BAJA' THEN 3
            END
        """)
 
        resultados = [dict(fila) for fila in cursor.fetchall()]
        conexion.close()
 
        return {"estadisticas_por_productividad": resultados}
 
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 5: LISTADO DE ESTUDIANTES (con paginación)
# GET /estudiantes
# =============================================================================
 
@app.get("/estudiantes", tags=["Estudiantes"])
def listar_estudiantes(
    pagina: int = Query(1, ge=1, description="Número de página"),
    limite: int = Query(10, ge=1, le=100, description="Registros por página (máx. 100)"),
    nivel: Optional[str] = Query(None, description="Filtrar por nivel: ALTA, MEDIA o BAJA")
):
    """Devuelve una lista paginada de estudiantes, con filtro opcional por nivel de productividad."""
    try:
        offset = (pagina - 1) * limite
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        if nivel:
            cursor.execute("""
                SELECT id, age, gender, study_hours, sleep_hours,
                       phone_usage, social_media, gaming_hours,
                       final_grade, productivity, nivel_productividad
                FROM estudiantes
                WHERE UPPER(nivel_productividad) = UPPER(?)
                LIMIT ? OFFSET ?
            """, (nivel, limite, offset))
            estudiantes = [dict(fila) for fila in cursor.fetchall()]
 
            cursor.execute("SELECT COUNT(*) FROM estudiantes WHERE UPPER(nivel_productividad) = UPPER(?)", (nivel,))
        else:
            cursor.execute("""
                SELECT id, age, gender, study_hours, sleep_hours,
                       phone_usage, social_media, gaming_hours,
                       final_grade, productivity, nivel_productividad
                FROM estudiantes
                LIMIT ? OFFSET ?
            """, (limite, offset))
            estudiantes = [dict(fila) for fila in cursor.fetchall()]
 
            cursor.execute("SELECT COUNT(*) FROM estudiantes")
 
        total = cursor.fetchone()[0]
        conexion.close()
 
        total_paginas = (total + limite - 1) // limite
 
        return {
            "paginacion": {
                "pagina_actual": pagina,
                "limite": limite,
                "total_registros": total,
                "total_paginas": total_paginas
            },
            "filtro_nivel": nivel if nivel else "Sin filtro",
            "estudiantes": estudiantes
        }
 
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 6: DATOS DE UN ESTUDIANTE POR ID
# GET /estudiante/{id}
# =============================================================================
 
@app.get("/estudiante/{id_estudiante}", tags=["Estudiantes"])
def obtener_estudiante(id_estudiante: int):
    """Devuelve todos los datos de un estudiante específico por su ID."""
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        cursor.execute("SELECT * FROM estudiantes WHERE id = ?", (id_estudiante,))
        fila = cursor.fetchone()
        conexion.close()
 
        if fila is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ningún estudiante con ID {id_estudiante}"
            )
 
        return {"estudiante": dict(fila)}
 
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 7: PREDECIR NIVEL DE PRODUCTIVIDAD
# POST /predecir
# =============================================================================
 
@app.post("/predecir", tags=["Predicción"])
def predecir_productividad(datos: DatosEstudiante):
    """
    Recibe los datos de un estudiante y devuelve la predicción del nivel
    de productividad (ALTA / MEDIA / BAJA) usando el modelo Decision Tree.
    """
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible. Ejecuta primero Proyecto_Principal.py")
 
    try:
        genero_cod = enc_gender.transform([datos.gender])[0]
 
        total_distraccion = (
            datos.phone_usage_hours +
            datos.social_media_hours +
            datos.youtube_hours +
            datos.gaming_hours
        )
        ratio_estudio = datos.study_hours_per_day / (total_distraccion + 0.001)
 
        entrada = np.array([[
            datos.age,
            genero_cod,
            datos.study_hours_per_day,
            datos.sleep_hours,
            datos.phone_usage_hours,
            datos.social_media_hours,
            datos.youtube_hours,
            datos.gaming_hours,
            datos.breaks_per_day,
            datos.coffee_intake_mg,
            datos.exercise_minutes,
            datos.assignments_completed,
            datos.attendance_percentage,
            datos.stress_level,
            datos.focus_score,
            datos.final_grade,
            total_distraccion,
            ratio_estudio
        ]])
 
        prediccion_cod = modelo.predict(entrada)[0]
        probabilidades = modelo.predict_proba(entrada)[0]
        prediccion_texto = enc_productividad.inverse_transform([prediccion_cod])[0]
        probabilidad_max = float(max(probabilidades))
 
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO predicciones_api
            (fecha, study_hours, sleep_hours, phone_usage, social_media,
             youtube_hours, gaming_hours, prediccion, probabilidad)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            datos.study_hours_per_day,
            datos.sleep_hours,
            datos.phone_usage_hours,
            datos.social_media_hours,
            datos.youtube_hours,
            datos.gaming_hours,
            prediccion_texto,
            round(probabilidad_max, 4)
        ))
        conexion.commit()
        id_prediccion = cursor.lastrowid
        conexion.close()
 
        return {
            "id_prediccion": id_prediccion,
            "prediccion": prediccion_texto,
            "probabilidad_confianza": f"{probabilidad_max * 100:.1f}%",
            "probabilidades_por_clase": {
                nombre: f"{prob * 100:.1f}%"
                for nombre, prob in zip(enc_productividad.classes_, probabilidades)
            },
            "resumen_datos": {
                "horas_estudio": datos.study_hours_per_day,
                "horas_sueno": datos.sleep_hours,
                "total_distraccion": round(total_distraccion, 2),
                "ratio_estudio_distraccion": round(ratio_estudio, 3)
            }
        }
 
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Valor inválido en los datos: {str(error)}")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 8: HISTORIAL DE PREDICCIONES
# GET /predicciones
# =============================================================================
 
@app.get("/predicciones", tags=["Predicción"])
def historial_predicciones():
    """Devuelve el historial de las últimas 50 predicciones realizadas."""
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        cursor.execute("""
            SELECT * FROM predicciones_api
            ORDER BY id DESC
            LIMIT 50
        """)
        predicciones = [dict(fila) for fila in cursor.fetchall()]
 
        cursor.execute("""
            SELECT prediccion, COUNT(*) AS total
            FROM predicciones_api
            GROUP BY prediccion
        """)
        resumen = {fila["prediccion"]: fila["total"] for fila in cursor.fetchall()}
 
        cursor.execute("SELECT COUNT(*) FROM predicciones_api")
        total = cursor.fetchone()[0]
        conexion.close()
 
        return {
            "total_predicciones_realizadas": total,
            "resumen_predicciones": resumen,
            "ultimas_predicciones": predicciones
        }
 
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 9: ACTUALIZAR ESTUDIANTE (PUT)
# PUT /estudiante/{id}
# =============================================================================
 
@app.put("/estudiante/{id_estudiante}", tags=["Estudiantes"])
def actualizar_estudiante(id_estudiante: int, datos: DatosActualizacion):
    """Actualiza parcialmente los datos de un estudiante existente."""
    try:
        campos_actualizar = {k: v for k, v in datos.dict().items() if v is not None}
 
        if not campos_actualizar:
            raise HTTPException(status_code=400, detail="No se enviaron campos para actualizar")
 
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        cursor.execute("SELECT id FROM estudiantes WHERE id = ?", (id_estudiante,))
        if cursor.fetchone() is None:
            conexion.close()
            raise HTTPException(status_code=404, detail=f"No existe el estudiante con ID {id_estudiante}")
 
        set_clausula = ", ".join([f"{campo} = ?" for campo in campos_actualizar.keys()])
        valores = list(campos_actualizar.values()) + [id_estudiante]
 
        cursor.execute(f"UPDATE estudiantes SET {set_clausula} WHERE id = ?", valores)
        conexion.commit()
        conexion.close()
 
        return {
            "mensaje": f"Estudiante con ID {id_estudiante} actualizado correctamente",
            "campos_actualizados": campos_actualizar
        }
 
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# ENDPOINT 10: ELIMINAR ESTUDIANTE (DELETE)
# DELETE /estudiante/{id}
# =============================================================================
 
@app.delete("/estudiante/{id_estudiante}", tags=["Estudiantes"])
def eliminar_estudiante(id_estudiante: int):
    """Elimina un estudiante de la base de datos por su ID."""
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
 
        cursor.execute("SELECT id FROM estudiantes WHERE id = ?", (id_estudiante,))
        if cursor.fetchone() is None:
            conexion.close()
            raise HTTPException(status_code=404, detail=f"No existe el estudiante con ID {id_estudiante}")
 
        cursor.execute("DELETE FROM estudiantes WHERE id = ?", (id_estudiante,))
        conexion.commit()
        conexion.close()
 
        return {"mensaje": f"Estudiante con ID {id_estudiante} eliminado correctamente"}
 
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
 
# =============================================================================
# ENDPOINT 11: CONSEJO PERSONALIZADO CON IA (Groq)
# POST /consejo
# =============================================================================
 
class DatosConsejo(BaseModel):
    nivel_productividad: str
    horas_estudio: float
    horas_sueno: float
    total_distraccion: float
    stress_level: int
 
    class Config:
        json_schema_extra = {
            "example": {
                "nivel_productividad": "BAJA",
                "horas_estudio": 2.0,
                "horas_sueno": 5.0,
                "total_distraccion": 6.0,
                "stress_level": 8
            }
        }
 
@app.post("/consejo", tags=["IA - Groq"])
def generar_consejo(datos: DatosConsejo):
    """
    Recibe el nivel de productividad y los hábitos del estudiante,
    y devuelve consejos personalizados generados por IA usando la API de Groq.
    Requiere la variable de entorno GROQ_API_KEY en el fichero .env
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="No se encontró la API Key de Groq. Asegúrate de tener el fichero .env con GROQ_API_KEY."
        )
 
    prompt = f"""
Eres un orientador académico experto. Un estudiante tiene los siguientes hábitos:
- Nivel de productividad predicho: {datos.nivel_productividad}
- Horas de estudio al día: {datos.horas_estudio}
- Horas de sueño: {datos.horas_sueno}
- Horas de distracción total (móvil, redes sociales, YouTube, gaming): {datos.total_distraccion}
- Nivel de estrés (1-10): {datos.stress_level}
 
Da exactamente 3 consejos concretos, prácticos y breves para mejorar su productividad.
Responde en español. Sé directo y motivador.
"""
 
    try:
        respuesta = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            },
            timeout=15
        )
 
        if respuesta.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Error al conectar con Groq: {respuesta.text}"
            )
 
        resultado = respuesta.json()
        consejo_texto = resultado["choices"][0]["message"]["content"]
 
        return {
            "nivel_productividad": datos.nivel_productividad,
            "consejos_ia": consejo_texto,
            "modelo_ia": "llama3-8b-8192",
            "proveedor": "Groq"
        }
 
    except http_requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado al conectar con Groq")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 55)
    print("ARRANCANDO SERVIDOR DE LA API")
    print("=" * 55)
    print("La API estará disponible en: http://127.0.0.1:5000")
    print("Documentación Swagger:       http://127.0.0.1:5000/docs")
    print("Pulsa CTRL+C para detener el servidor")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=5000)
