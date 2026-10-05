# =============================================================================
# DOCKERFILE - Student Productivity Prediction API
# =============================================================================
# Para construir la imagen:
#   docker build -t student-productivity-api .
#
# Para ejecutar el contenedor:
#   docker run -p 5000:5000 student-productivity-api
#
# La API estará disponible en http://localhost:5000
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="Estudiante ESIC University"
LABEL description="API de predicción de productividad estudiantil"
LABEL version="1.0"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python Proyecto_Principal.py

EXPOSE 5000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
