import dataCollection
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

data = dataCollection.load_data()

# Inicializar diccionario para acumular el tiempo y contar las ocurrencias
time_totals = defaultdict(list)
counts = defaultdict(int)

# Filtrar datos para excluir "DFSL"
filtered_data = [row for row in data if row['algorithm_name'] != 'DFSL' and row['solution_found'] == 'True']

# Acumular los tiempos por algoritmo
for row in filtered_data:
    algorithm = row['algorithm_name']
    time_totals[algorithm].append(float(row['time']) * 1000)
    counts[algorithm] += 1

# Calcular los promedios y desviaciones estándar
averages = {alg: np.mean(time_totals[alg]) for alg in time_totals}
std_devs = {alg: np.std(time_totals[alg]) for alg in time_totals}

# Datos para el gráfico
algorithms = list(averages.keys())
average_time = [averages[alg] for alg in algorithms]
std_time = [std_devs[alg] for alg in algorithms]

# Crear la figura y el eje para el gráfico de tiempo
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de barras para el tiempo con barras de error
ax[0].bar(algorithms, average_time, yerr=std_time, color='salmon', capsize=5)
ax[0].set_title('Tiempo promedio por Algoritmo')
ax[0].set_xlabel('Algoritmo')
ax[0].set_ylabel('Tiempo (milisegundos)')

# Gráfico de caja y bigotes (boxplot)
ax[1].boxplot([time_totals[alg] for alg in algorithms], labels=algorithms)
ax[1].set_title('Distribución de tiempo por Algoritmo')
ax[1].set_xlabel('Algoritmo')
ax[1].set_ylabel('Tiempo (milisegundos)')

# Ajustar el diseño y mostrar la gráfica
plt.tight_layout()
plt.show()
