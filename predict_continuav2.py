import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
import pytesseract
import time
import base64
import requests
#pantalla
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306, ssd1325, ssd1331, sh1106
from time import sleep
import socket

#decodigo.com
serial = i2c(port=1, address=0x3C)
#device = ssd1306(serial, rotate=0)
device = sh1106(serial, width=128, height=64, rotate=0)


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
        #LECTURA TESSERACT OCR
                
        #convertir a gris la imagen
        g = cv2.cvtColor(region_recortada, cv2.COLOR_BGR2GRAY)

        # Crear un elemento estructurante circular con radio 1 utilizando OpenCV
        conc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

        # Dilatar 'g' utilizando el elemento estructurante circular
        gi = cv2.dilate(g, conc)
        cv2.imshow('gi', gi)

        options = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # use Tesseract to OCR the image
        placa = pytesseract.image_to_string(gi, config=options)
          
        placa = placa.strip()
        placa = placa.replace("\n", "")
        placa = placa.replace("\r", "")
        placa = placa.replace(" ", "")
    except Exception as e:
        # Manejo de la excepción, por ejemplo, mostrar el mensaje de error
        print("Error en OCR:", e)
        placa=""
        
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        draw.text((10, 0), "iHUB UNHEVAL", fill="white")
      
        draw.text((10, 20), "PLACA:" + placa, fill="white") 
    
    
  
    
    #ENVIAR PLACA POR SERVICIO:
    
    if len(placa) > 0:
        
          
        
        if len(placa) >= 6:
         placa = placa[:6]
        
        imagen_base64 = image_to_base64(imagen)

        # Datos a enviar por POST
        data = {
            "id_usuario": "3",
            "codigo_estacion": "002",
            "placa": placa,
            "img": imagen_base64
        }

        # URL del endpoint donde deseas enviar los datos
        url = "http://192.168.1.55:90/app_matri/registro_lectura.php"
        
        
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
    cap = cv2.VideoCapture(0)
 
    while True:
        
        c+=1
        
       
        #realizar inferencia
        if(cap.isOpened()):
        
            # Leer el frame desde la cámara
            ret, frame = cap.read()

            if not ret:
                break

            # Convertir el frame a formato RGB utilizando OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convertir el frame a formato PIL.Image
            preprocessed_image = Image.fromarray(frame_rgb)

            # Predecir usando el modelo
            predictions = od_model.predict_image(preprocessed_image)
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
            cv2.imshow('Frame', frame)

            # Limpiar la memoria de video de OpenCV
            cv2.waitKey(1)
            
            time.sleep(1)  # espera 2 segundos entre cada print()
            
            
        # Detener la ejecución si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
