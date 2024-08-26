import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = r'C:\Users\joaqu\Desktop\Análisis agentes racionales.xlsx'
sheet_name = "128x128"
df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

valores_x = ["0.1", "0.2", "0.4", "0.8"]

# Extracción de datos, puntaje de performance
performance_simple = df.iloc[2, 1:41]
performance_random = df.iloc[3, 1:41]

unique_x = valores_x

# Separar los datos en grupos de 10 para cada valor de x
grouped_simple = [performance_simple[i*10:(i+1)*10] for i in range(4)]
grouped_random = [performance_random[i*10:(i+1)*10] for i in range(4)]

# Obtener el valor máximo de performance_simple y performance_random
max_performance_simple = performance_simple.max()
max_performance_random = performance_random.max()

# Determinar el valor máximo a usar para el límite del eje y
max_performance = max(max_performance_simple, max_performance_random)

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Graficar performance simple con marcador circular
for i in range(4):
    plt.scatter([unique_x[i]]*10, grouped_simple[i], color='blue', marker='o', label='Agente reflexivo simple' if i == 0 else "")

# Graficar performance random con marcador "x"
for i in range(4):
    plt.scatter([unique_x[i]]*10, grouped_random[i], color='red', marker='x', label='Agente aleatorio' if i == 0 else "")

# Añadir etiquetas y título
plt.xlabel('Tasa de suciedad')
plt.ylabel('Puntaje de performance')
plt.title('Comparación de performance de agente reflexivo simple y random: Entorno ' + sheet_name)

# Ajustar los límites del eje y para que comiencen en 0 y terminen en el valor máximo + un margen
plt.ylim(0, max_performance * 1.1)  # Añadir un 10% de margen superior

# Ajustar los ticks del eje y para que haya 10 ticks uniformemente distribuidos
y_min, y_max = plt.ylim()
ticks = range(0, int(y_max) + 1, max(1, int((y_max - y_min) / 10)))
plt.yticks(ticks)

# Añadir leyenda
plt.legend()

# Mostrar la gráfica
plt.show()
