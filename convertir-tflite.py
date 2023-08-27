import tensorflow as tf

# Cargar el modelo TensorFlow existente
saved_model_path = '/home/charlen/Dropbox/COMPARTIR_LINUX/PROGRAMACION/RASPBERRY/proy_placas/v5/model.pb'
model = tf.saved_model.load(saved_model_path)

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

# Guardar el modelo TensorFlow Lite en un archivo
with open('/home/charlen/Dropbox/COMPARTIR_LINUX/PROGRAMACION/RASPBERRY/proy_placas/v5/modelo.tflite', 'wb') as f:
    f.write(tflite_model)
