import easyocr
import sys
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

import time
import base64
import requests
import os

from object_detectionv2 import ObjectDetection

# Definir el idioma del texto que deseas reconocer
idioma = 'en'  # 'es' para español, 'en' para inglés, etc. Consulta la documentación para más idiomas.

# Cargar el modelo de EasyOCR para el idioma específico
lector = easyocr.Reader([idioma], gpu=False)

# Configurar el modelo para que excluya símbolos y conserve letras y números
listablanca = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]

def dibujar_cuadro(imagen, bounding_box, codi, color=(0, 255, 0), grosor=2):
    height, width, _ = imagen.shape
    left = int(bounding_box['left'] * width)
    top = int(bounding_box['top'] * height)
    right = int((bounding_box['left'] + bounding_box['width']) * width)
    bottom = int((bounding_box['top'] + bounding_box['height']) * height)

    # Sin cambios
    cv2.imwrite('original/o_' + str(codi) + '.jpg', imagen)
    cv2.rectangle(imagen, (left, top), (right, bottom), color, grosor)

    cv2.imshow('captura', imagen)

    # Con cuadro
    cv2.imwrite('inferencia/i_' + str(codi) + '.jpg', imagen)

    region_recortada = imagen[top:bottom, left:right]

    placa = ''

    try:
        # LECTURA TESSERACT OCR
        resultado = lector.readtext(region_recortada, allowlist=listablanca)
        for res in resultado:
            placa = (res[1])

            print("########## PLACA #########")
            print(res[1])
            print("##########################")

    except Exception as e:
        # Manejo de la excepción, por ejemplo, mostrar el mensaje de error
        print("Error en OCR:", e)
        placa = "-"

    # ENVIAR PLACA POR SERVICIO:

    if len(placa) > 0:

        if len(placa) >= 6:
            placa = placa[:6]

        imagen_base64 = image_to_base64(imagen)

        # Datos a enviar por POST
        data = {
            "id_usuario": "1",
            "codigo_estacion": "E001",
            "placa": placa,
            "img": imagen_base64
        }

        # URL local
        # url = "http://192.168.1.55:90/app_matri/registro_lectura.php"
        # URL web
        url = "https://deteccionplaca.000webhostapp.com/registro_lectura.php"

        try:
            # Realizar la solicitud POST con los datos y la imagen en base64
            response = requests.post(url, json=data)

            # Verificar la respuesta
            if response.status_code == 200:
                print("La placa se envió correctamente.")
            else:
                print("Hubo un error al enviar la placa.")

        except Exception as e:
            print("Error en API:", e)

    else:
        # Realizar otra acción si la placa no tiene 6 caracteres
        print("La placa no detectada")
        # Agrega aquí el código para realizar otra acción si es necesario
        # Convertir la imagen a base64


def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer).decode("utf-8")
    return encoded_string


def capturar_imagen(nombre_archivo):
    # Usar fswebcam para capturar una imagen desde la cámara
    os.system(f'fswebcam --device /dev/video1 -r 640x320 --no-banner {nombre_archivo}')
    #fswebcam --device /dev/video0 image.jpg

imagen_temporal = "imagen_temp.jpg"


def main():
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

    c = 0
    
    while True:
        # Capturar la imagen utilizando fswebcam
        capturar_imagen(imagen_temporal)

        # Leer la imagen desde el archivo
        frame = cv2.imread(imagen_temporal)
     
        # Convertir el frame a formato RGB utilizando OpenCV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir el frame a formato PIL.Image
        preprocessed_image = Image.open(imagen_temporal)

        # Predecir usando el modelo
        predictions = od_model.predict_image(preprocessed_image)
        print(predictions)
            
           # predictions = tf_model.predict_image(preprocessed_image)
            # Encontrar el resultado con la probabilidad máxima
        max_prob_result = None
        max_prob = 0

        for resultado in predictions:
            probabilidad = resultado['probability']
            if probabilidad > max_prob:
                max_prob = probabilidad
                max_prob_result = resultado

            # Si se encontró un resultado con probabilidad mayor a 0.80, dibujar el cuadro delimitador correspondiente
        if max_prob_result is not None:
            dibujar_cuadro(frame, max_prob_result['boundingBox'],c)
        # Mostrar el frame con el cuadro delimitador si se encontró un resultado con probabilidad mayor a 0.80
        #cv2.imshow('Frame', frame)

        # Limpiar la memoria de video de OpenCV
        cv2.waitKey(1)

        # Detener la ejecución si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
