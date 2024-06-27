import torch
import numpy as np
import matplotlib.pyplot as plt

#Crear clase del modelo AND del perceptron
class PerceptronAND(torch.nn.Module):
    def __init__(self):
        super(PerceptronAND, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

#Definir datos de entrenamiento
x_train = torch.from_numpy(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).float()
y_train = torch.from_numpy(np.array([0, 0, 0, 1])).float()

y_train = y_train.reshape(-1, 1)

#Crear el modelo y el optimizador
model = PerceptronAND()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

# Inicializar listas para la pérdida (Gráfico)
loss_values = []
epoch_values = []

#Entrenar el modelo (creación de bucle)
print('Comenzando entrenamiento...')
for epoch in range(100):
  #Predecir la salida
  y_pred = model(x_train)

  #Calcular la pérdida
  loss = torch.nn.BCELoss()(y_pred, y_train)

  #Gradiente cero
  optimizer.zero_grad()

  #Calcular el gradiente
  loss.backward()

  #Actualizar los pesos
  optimizer.step()
  
  # Almacenar valores para el gráfico
  loss_values.append(loss.item())
  epoch_values.append(epoch)

  #Imprimir la pérdida
  if epoch % 10 == 0:
    print('Época:', epoch, 'Pérdida:', loss.item())
print('Modelo entrenado...')

# Crear el gráfico
plt.plot(epoch_values, loss_values, 'b-o', label='Pérdida')
plt.title('Perdida durante el entrenamiento del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True)
    
#Evaluar el modelo en datos de prueba
y_pred = model(x_train)

#Imprimir las predicciones
print("Entradas:\n", x_train)
print("Salidas:", y_pred)

#Redondear las salidas predichas a 0 o 1
y_pred = torch.round(y_pred)
print("Salidas:\n", y_pred)

# Obtener los valores de w1 y w2
linear_layer = model.linear
print(f'w1: {w1 = linear_layer.weight[0, 0].item()}')
w2 = linear_layer.weight[0, 1].item()

# Imprimir los valores de w1 y w2
print("w1:", w1)
print("w2:", w2)
