from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

#Obtener las variables de datos 'train' y 'test'
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

#Crear etiquetas de texto para posibles respuestas
class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro',
    'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve'
]

#normaliar los datos (pasar de 0-255 a 0-1)
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 #0 a
    return imagenes, etiquetas

#Normalizar los datos de entrenamiento
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#crear modelo de entrenamiento
modelo = tf.keras.Sequential([
    #Capas ocultas
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    #Capa de salida
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),   
])

#Compilar modelo de entrenamiento
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Buscar la cantidad de números de entrenamiento y pruebas
num_ejemplos_entrenamiento = metadatos.splits['train'].num_examples
num_ejemplos_pruebas = metadatos.splits['test'].num_examples

print(num_ejemplos_entrenamiento, num_ejemplos_pruebas)

#Optimizar entrenamiento con lotes
TAMANO_LOTE = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ejemplos_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

#Entrenar modelo
historial = modelo.fit(
    datos_entrenamiento, epochs = 5,
    steps_per_epoch = math.ceil(num_ejemplos_entrenamiento/TAMANO_LOTE)
)

#Gráfico de perdida
plt.xlabel("# Epoca")
plt.ylabel("Pérdida")
plt.plot(historial.history["loss"])
plt.show()

#Evaluar el modelo ya entrenado
test_loss, test_accuracy = modelo.evaluate(
	datos_pruebas, steps=math.ceil(num_ejemplos_pruebas/32)
)

print("Precisión en las pruebas: ", test_accuracy)

#Gráficar predicciones con: Azul si es correcta, rojo si es falsa.
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("Prediccion: {}".format(class_names[etiqueta_prediccion]), color=color)

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#888888")
  plt.ylim([0, 1])
  etiqueta_prediccion = np.argmax(arr_predicciones)

  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')

filas = 5
columnas = 3
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
  
plt.show()
