from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import os

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Ruta de la carpeta de datos
directorio_datos = "optimized_data/extracted_images"

#Cantidad de carpetas con datos(se redujo los 82 originales por los tiempos de carga)
# Contar la cantidad de carpetas (clases) en el directorio de datos
cantidad_datos = len([nombre for nombre in os.listdir(directorio_datos) if os.path.isdir(os.path.join(directorio_datos, nombre))])

print(f"Cantidad de carpetas (clases) encontradas: {cantidad_datos}")

# Crear un generador de datos
generador_datos = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2    
)

# Crear los conjuntos de entrenamiento y prueba
datos_entrenamiento = generador_datos.flow_from_directory(
    directorio_datos,
    target_size = (45, 45),
    batch_size = 32,
    class_mode='categorical',
    subset = 'training'   
)

datos_validacion = generador_datos.flow_from_directory(
    directorio_datos,
    target_size = (45, 45),
    batch_size = 32,
    class_mode='categorical',
    subset = 'validation'
)

# Obtener los nombres de las carpetas, que funcionan como nombre de los carácteres tambien
categorias_clases = list(datos_entrenamiento.class_indices.keys())
print("Nombres de carácteres:", categorias_clases)

# crear modelo de entrenamiento
modelo = tf.keras.Sequential([
    #Capas ocultas
    tf.keras.layers.Flatten(input_shape=(45, 45, 3)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    #Capa de salida
    tf.keras.layers.Dense(cantidad_datos, activation=tf.nn.softmax),   
])

# Compilar modelo de entrenamiento
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Buscar la cantidad de números de entrenamiento y validación
num_imagenes_entrenamiento = datos_entrenamiento.samples
num_imagenes_validacion = datos_validacion.samples

print("Número de imágenes para entrenamiento:", num_imagenes_entrenamiento)
print("Número de imágenes para validación:", num_imagenes_validacion)

# Entrenar modelo
historial = modelo.fit(
    datos_entrenamiento,
    epochs = 5,
    steps_per_epoch = math.ceil(num_imagenes_entrenamiento)
)

# Gráfico de perdida
plt.xlabel("# Epoca")
plt.ylabel("Pérdida")
plt.plot(historial.history["loss"])
plt.show()

# Evaluar el modelo ya entrenado
test_loss, test_accuracy = modelo.evaluate(
	datos_validacion, steps=math.ceil(num_imagenes_validacion/32)
)

print("Precisión en las pruebas: ", test_accuracy)

# Obtener un lote de datos de prueba
imagenes_prueba, etiquetas_prueba  = next(iter(datos_validacion))

# Predecir en el lote de imágenes de prueba
predicciones = modelo.predict(imagenes_prueba)

# Gráficar predicciones con: Azul si es correcta, rojo si es falsa.
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  etiqueta_real = np.argmax(etiqueta_real)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("Prediccion: {}".format(categorias_clases[etiqueta_prediccion]), color=color)

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(cantidad_datos), arr_predicciones, color="#888888")
  plt.ylim([0, 1])
  etiqueta_prediccion = np.argmax(arr_predicciones)
  etiqueta_real = np.argmax(etiqueta_real)

  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')

# Gráfico de resultados
filas = 5
columnas = 3
num_imagenes = filas * columnas
plt.figure(figsize=(2 * 2 * columnas, 2 * filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2 * columnas, 2 * i + 1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2 * columnas, 2 * i + 2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)

plt.show()
