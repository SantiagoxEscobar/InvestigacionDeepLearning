import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Definir la tabla de valores: input(x1, x2) --> output (y)  
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

#Definir el modelo mediante Keras (redes neuronales)
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(units = 1, input_shape = [2], activation='sigmoid')
)

#Compilar el modelo para ser entrenado
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

#Entrenando el modelo
print('Comenzando entrenamiento...')
epochs_hist = model.fit(x_train, y_train, epochs = 110, verbose = False)
print('Modelo entrenado...')

#Evaluación del modelo
epochs_hist.history.keys()

#Gráfico de pérdida durante entrenamiento
plt.plot(epochs_hist.history['loss'])
plt.title('Perdida durante el entrenamiento del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True)

#Prueba del perceptron entrenado
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_resultado = model.predict(x_test)
y_resultado2 = tf.where(model.predict(x_test) >= 0.5, 1, 0)

print("Entradas:\n", x_test)
print()

print("Salidas:\n", y_resultado)
print()

print("Salidas redondeadas:", y_resultado2)
print()

#Mostrar los pesos w1 y w2
pesos_capa = model.layers[0].get_weights()
w = pesos_capa[0]
print("Pesos w1 y w2:", w)
