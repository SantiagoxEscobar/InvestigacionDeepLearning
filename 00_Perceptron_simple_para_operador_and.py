import numpy as np
import matplotlib.pyplot as plt

#Definir la tabla de valores: input(x1, x2) --> output (y)  
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

#Definir los valores aleatorios entre 0 y 1 a los pesos y al bias
w1 = np.random.rand()
w2 = np.random.rand()
bias = np.random.rand()
learn = np.random.rand()

#Funcion de la suma ponderada
def perceptron_and(x1, x2, w1, w2, bias):
    z = x1 * w1 + x2 * w2 + bias
    
    #Condicion de activación
    if z > 0:
        return 1
    else:
        return 0

#Creacion de bucle de entrenamiento
for epoch in range(1000):
    contador_errores = 0
    
    for i in range(len(x)):
        x1 = x[i][0]
        x2 = x[i][1]
        y_pred = perceptron_and(x1, x2, w1, w2, bias)      
        
        #Contar los errores que validan en 0
        error = y[i] - y_pred
        if error == 0:
            contador_errores += 1
        else:
            #Actualizar los pesos      
            w1_aux = learn * error * x1
            w1 += w1_aux
            w2_aux = learn * error * x2
            w2 += w2_aux
            bias_aux = learn * error
            bias += bias_aux
        
    #Si todas las filas dan 0 se sale del ciclo (Neurona entrenada)
    if contador_errores == len(x):
        break

#Prueba del perceptron entrenado
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = np.array([perceptron_and(x1, x2, w1, w2, bias) for x1, x2 in x_test])

print("Entradas:", x_test)
print("Salidas:", y_pred)
print(f'W1: {w1}, W2: {w2}')

#Graficar la recta del perceptron
plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.scatter(x[:, 0], x[:, 1], c='red')

x1 = np.arange(0, 1.5, 0.1)
x2 = (-bias - x1 * w1) / w2
plt.plot(x1, x2, label='Recta')
plt.xlim([-0.1, 1.5])
plt.ylim([-0.1, 1.5])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Función lineal - Perceptron AND')
plt.legend()
plt.show()
