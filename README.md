# Identificación de personas por caminata usando landmarks corporales 🚶‍♀️

Este proyecto forma parte de la Maestría en Ciencias de la Computación y
consiste en desarrollar un sistema para la identificación de personas a
partir de secuencias de caminata, utilizando landmarks corporales obtenidos
de videos.

Se trabajó con un conjunto de datos colectivo de secuencias grabadas por
diferentes personas, utilizando un conjunto para entrenamiento y un
conjunto independiente para validación externa.

## 🧠 Descripción

El objetivo principal es transformar secuencias de pose corporal en
descriptores numéricos y entrenar modelos de clasificación para identificar
a qué persona pertenece cada secuencia de caminata.

Durante el desarrollo se realizaron los siguientes pasos principales:

- Extracción de landmarks corporales por frame utilizando MediaPipe Pose.
- Construcción de archivos CSV con las coordenadas 3D de cada articulación.
- Extracción de características estadísticas por secuencia.
- Escalado de las características.
- Entrenamiento de distintos modelos de clasificación.
- Evaluación mediante validación cruzada y análisis cualitativo de errores.

## 🛠️ Modelos utilizados

Se evaluaron distintos modelos, seleccionados por su simplicidad y buen
desempeño en conjuntos de datos pequeños:

- **Linear SVM (SVM lineal)**: clasificador lineal adecuado para espacios de
  alta dimensión.
- **Random Forest**: modelo basado en ensambles de árboles de decisión.
- **KNN (K-Nearest Neighbors)**: clasificador basado en vecinos más cercanos.

El modelo SVM lineal presentó el mejor desempeño promedio en términos de
F1-score macro durante la validación cruzada, por lo que fue seleccionado
como modelo final.

## 📊 Resultado general

Los modelos fueron evaluados mediante validación cruzada sobre el conjunto
de entrenamiento. Posteriormente, el modelo final se aplicó a un conjunto de
prueba independiente para generar las predicciones.

A partir de la revisión manual de los videos de prueba se observó que muchos
errores están asociados a movimientos adicionales durante la caminata
(sacar el teléfono, consultar el reloj, manipular objetos), así como a
inestabilidad en la detección de la pose, principalmente en brazos, manos y
hombros. Estas condiciones afectan directamente los descriptores
calculados y explican parte de las confusiones observadas.

## 📂 Organización del repositorio

Este repositorio incluye los siguientes archivos principales:

- `Sensado_análisis_video.ipynb`  
  Notebook principal con todo el flujo de análisis, entrenamiento y
  evaluación.

- `batch_extract_pose.py`  
  Script utilizado para la extracción de landmarks corporales a partir de
  los videos y generación de los archivos CSV de pose.

- `labels_example.csv`  
  Archivo de ejemplo con el formato de etiquetas utilizado por el notebook.
  Contiene únicamente identificadores numéricos y se incluye como plantilla.

> Para ejecutar el notebook, renombra `labels_example.csv` a `labels.csv`.

El archivo real de etiquetas utilizado en el experimento no se incluye en
este repositorio.

## 📌 Nota sobre el modelo de MediaPipe

El archivo del modelo de MediaPipe Pose (`pose_landmarker.task`) no se
incluye en el repositorio debido a restricciones de tamaño.

Para poder ejecutar el script de extracción de pose, es necesario descargar
el modelo oficial de MediaPipe Pose y colocar el archivo
`pose_landmarker.task` en la misma carpeta que el script
`batch_extract_pose.py`.

## 📁 Requisitos

Para ejecutar el notebook correctamente es necesario contar con:

- Python 3.x
- Librerías principales:
  - `mediapipe`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `opencv-python`

Si se desea reproducir el entorno completo utilizado en la práctica, se
incluye el archivo `environment.yml` para crear el entorno con conda.

