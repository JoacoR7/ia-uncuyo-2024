import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = r'C:\Users\joaqu\Desktop\pruebas.xlsx'
df = pd.read_excel(file_path, sheet_name='128x128', header=None)

# Contar el número de filas en el DataFrame
row_count = df.shape[0] - 2

# Reemplazar "-" con NaN
df.replace("-", np.nan, inplace=True)

# Extraer los datos de performance
performance_simple = df.iloc[4:(int(row_count/2)+3), 31:41]
performance_random = df.iloc[3+int(row_count/2):row_count+2, 31:41]

def performance_mean(df_info):
    performance_mean_list = []
    for index, row in df_info.iterrows():
        sum = 0
        count = 0
        mean = 0
        for col in df_info.columns:
            if np.isnan(row[col]):
                pass
            else:
                sum += row[col]
                count += 1
        if sum != 0:
            mean = sum / count
        performance_mean_list.append(mean)
    return performance_mean_list

performance_simple_mean = performance_mean(performance_simple)
performance_random_mean = performance_mean(performance_random)

# Crear gráfico de barras con ambos conjuntos de datos
bar_width = 0.4  # Ancho de las barras
x_positions = np.arange(len(performance_simple_mean))  # Posiciones en el eje X

# Ajustar el tamaño de la figura horizontalmente
plt.figure(figsize=(20, 10))

# Graficar barras para 'performance_simple_mean'
plt.bar(x_positions - bar_width/2, performance_simple_mean, width=bar_width, color='blue', label='Agente reflexivo simple')

# Graficar barras para 'performance_random_mean'
plt.bar(x_positions + bar_width/2, performance_random_mean, width=bar_width, color='green', label='Agente random')

# Ajustar las posiciones de las etiquetas del eje X para que coincidan con las barras
# Mostrar cada 10 valores más del límite actual
x_limit = len(performance_simple_mean) + 10
plt.xticks(np.arange(0, x_limit, 10), [str(i+1) for i in np.arange(0, x_limit, 10)])

# Ajustar el eje Y para que vaya de 0 a 1000
plt.ylim(0, 1000)

# Establecer los pasos del eje Y cada 50
plt.yticks(np.arange(0, 1001, 50))

# Etiquetas, título y leyenda
plt.xlabel('Puntos de performance')
plt.ylabel('Cantidad de acciones')
plt.title('Comparación de cantidad de acciones tomadas para llegar a cada punto de performance')
plt.legend()

# Mostrar el gráfico
plt.show()
