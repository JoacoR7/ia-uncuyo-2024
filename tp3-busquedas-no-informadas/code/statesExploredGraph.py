import dataCollection
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

data = dataCollection.load_data()

# Inicializar diccionarios para acumular los estados y contar las ocurrencias
states_totals = defaultdict(list)
counts = defaultdict(int)

# Filtrar datos para excluir "DFSL"
filtered_data = [row for row in data if row['algorithm_name'] != 'DFSL' and row['solution_found'] == 'True']

# Acumular los valores por algoritmo
for row in filtered_data:
    algorithm = row['algorithm_name']
    states_totals[algorithm].append(int(row['states_n']))
    counts[algorithm] += 1

# Calcular los promedios y desviaciones estándar
averages = {alg: np.mean(states_totals[alg]) for alg in states_totals}
std_devs = {alg: np.std(states_totals[alg]) for alg in states_totals}

# Datos para el gráfico
algorithms = list(averages.keys())
average_states_n = [averages[alg] for alg in algorithms]
std_states_n = [std_devs[alg] for alg in algorithms]

# Crear la figura y el eje para el gráfico de estados visitados
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de barras para States N con barras de error
ax[0].bar(algorithms, average_states_n, yerr=std_states_n, color='salmon', capsize=5)
ax[0].set_title('Promedio de estados visitados')
ax[0].set_xlabel('Algoritmo')
ax[0].set_ylabel('Cantidad de estados')

# Gráfico de caja y bigotes (boxplot)
ax[1].boxplot([states_totals[alg] for alg in algorithms], labels=algorithms)
ax[1].set_title('Distribución de estados visitados')
ax[1].set_xlabel('Algoritmo')
ax[1].set_ylabel('Cantidad de estados')

# Ajustar el diseño y mostrar la gráfica
plt.tight_layout()
plt.show()
