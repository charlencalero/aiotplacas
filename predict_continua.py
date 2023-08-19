import easyocr
import sys
import os
# Configurar CUDA_VISIBLE_DEVICES para forzar la ejecución en la CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
import time
import base64
import requests
#pantalla

from time import sleep
import socket
import datetime


# Definir el idioma del texto que deseas reconocer
idioma = 'en'  # 'es' para español, 'en' para inglés, etc. Consulta la documentación para más idiomas.

# Cargar el modelo de EasyOCR para el idioma específico
lector = easyocr.Reader([idioma])


MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'



class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float64)[:, :, (2, 1, 0)]  # RGB -> BGR
        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]




def dibujar_cuadro(imagen, bounding_box,codi, color=(0, 255, 0), grosor=2):
    height, width, _ = imagen.shape
    left = int(bounding_box['left'] * width)
    top = int(bounding_box['top'] * height)
    right = int((bounding_box['left'] + bounding_box['width']) * width)
    bottom = int((bounding_box['top'] + bounding_box['height']) * height)
    
    #sin cambios
    cv2.imwrite('original/o_'+ str(codi) +'.jpg', imagen)
    cv2.rectangle(imagen, (left, top), (right, bottom), color, grosor)
    
    #con cuadro
    cv2.imwrite('inferencia/i_'+ str(codi) +'.jpg', imagen)
       
    region_recortada = imagen[top:bottom, left:right]
    
    placa=''
    
    try:
       resultado = lector.readtext(region_recortada)
       cv2.imshow('placa', region_recortada)
       for res in resultado:
        placa = res[1]
        print("######### placa ##########")
        print(res[1])
        print("##########################")
        
    except Exception as e:
       # Manejo de la excepción, por ejemplo, mostrar el mensaje de error
       print("Error en OCR:", e)
       placa = "-"
       
		
    #ENVIAR PLACA POR SERVICIO:
    
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
    os.system(f'fswebcam -r 1280x720 --no-banner {nombre_archivo}')
    
imagen_temporal = "imagen_temp.jpg"


def main():
    # Load a TensorFlow model
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)
 
        
    c=0

 
    while True:
        
            c=c+1
            
            print('***INICIO***')
            print(datetime.datetime.now().time())
            
            # Capturar la imagen utilizando fswebcam
            capturar_imagen(imagen_temporal)
            
            print('***CAPTURA***')
            print(datetime.datetime.now().time())
            
            # Leer la imagen desde el archivo
            frame = cv2.imread(imagen_temporal)
      
            # Convertir el frame a formato RGB utilizando OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convertir el frame a formato PIL.Image
            preprocessed_image = Image.fromarray(frame_rgb)

            # Predecir usando el modelo
            predictions = od_model.predict_image(preprocessed_image)
            
            print('***PREDICCION***')
            print(datetime.datetime.now().time())
            
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
            print('***FIN***')
            print(datetime.datetime.now().time())
            
            
            # Detener la ejecución si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
