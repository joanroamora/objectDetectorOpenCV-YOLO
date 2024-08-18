# 🇺🇸 Object Detector Project Documentation

## Introduction
This project is an Object Detection application using the YOLO (You Only Look Once) algorithm. It is designed to detect objects in images by leveraging the power of OpenCV and NumPy. The program processes an input image and identifies the object with the highest confidence level based on the YOLO model.

## How It Works

### Loading the YOLO Model:
- The project loads the pre-trained YOLO model, which consists of yolov3.weights and yolov3.cfg. 

- These files are essential for the model's functionality, containing the weights and configuration, respectively. However, due to their large size, they are not included in this repository.

- Note: The yolov3.weights file needs to be downloaded separately from the following repository: https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights

- The project also loads the coco.names file, which contains the list of classes the model can detect.

### Image Processing:
- The image is read and prepared for YOLO processing. This involves resizing and normalizing the image into a blob format that the YOLO model can work with.

### Object Detection:
- The program passes the image blob through the YOLO model, which returns potential objects detected in the image, along with their confidence scores and bounding boxes.

### Class Identification:
- The program identifies the class of the object with the highest confidence score and prints it as the detected object.

## Libraries Used
- **OpenCV**: For image processing and handling the YOLO model.
- **NumPy**: For numerical operations, particularly when processing the model's output.

## How to Use the Project

### Install the Required Libraries:
1. Ensure you have Python installed.
2. Install the necessary dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

### Run the Object Detection:
Use the following command in your terminal to run the object detection on an image:

```bash
python objectDetector.py <image path>
```

# 🇨🇴 Documentación del Proyecto de Detección de Objetos

## Introducción
Este proyecto es una aplicación de Detección de Objetos utilizando el algoritmo YOLO (You Only Look Once). Está diseñado para detectar objetos en imágenes utilizando el poder de OpenCV y NumPy. El programa procesa una imagen de entrada e identifica el objeto con el nivel de confianza más alto basado en el modelo YOLO.

## Cómo Funciona

### Carga del Modelo YOLO:
- El proyecto carga el modelo YOLO preentrenado, que consta de yolov3.weights y yolov3.cfg. 

- Estos archivos son esenciales para el funcionamiento del modelo, ya que contienen los pesos y la configuración, respectivamente. Sin embargo, debido a su gran tamaño, no se incluyen en este repositorio.

- Nota: El archivo yolov3.weights debe descargarse por separado desde el siguiente repositorio: https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights

- El proyecto también carga el archivo coco.names, que contiene la lista de clases que el modelo puede detectar

### Procesamiento de la Imagen:
- La imagen se lee y se prepara para el procesamiento con YOLO. Esto implica redimensionar y normalizar la imagen en un formato de blob que el modelo YOLO puede utilizar.

### Detección de Objetos:
- El programa pasa el blob de la imagen a través del modelo YOLO, que devuelve los posibles objetos detectados en la imagen, junto con sus puntajes de confianza y cuadros delimitadores.

### Identificación de Clases:
- El programa identifica la clase del objeto con el puntaje de confianza más alto y la imprime como el objeto detectado.

## Librerías Utilizadas
- **OpenCV**: Para procesamiento de imágenes y manejo del modelo YOLO.
- **NumPy**: Para operaciones numéricas, particularmente al procesar la salida del modelo.

## Cómo Usar el Proyecto

### Instalar las Librerías Necesarias:
1. Asegúrese de tener Python instalado.
2. Instale las dependencias necesarias ejecutando:

    ```bash
    pip install -r requirements.txt
    ```

### Ejecutar la Detección de Objetos:
Use el siguiente comando en su terminal para ejecutar la detección de objetos en una imagen:

```bash
python objectDetector.py <ruta de la imagen>
```