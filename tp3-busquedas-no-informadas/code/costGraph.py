import dataCollection
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

data = dataCollection.load_data()

# Inicializar diccionarios para acumular los costos y contar las ocurrencias
cost_e1_totals = defaultdict(list)
cost_e2_totals = defaultdict(list)
counts = defaultdict(int)

# Filtrar datos para excluir "DFSL"
filtered_data = [row for row in data if row['algorithm_name'] != 'DFSL' and row['solution_found'] == 'True']

# Acumular los valores por algoritmo
for row in filtered_data:
    algorithm = row['algorithm_name']
    cost_e1_totals[algorithm].append(float(row['cost_e1']))
    cost_e2_totals[algorithm].append(float(row['cost_e2']))
    counts[algorithm] += 1

# Calcular los promedios y desviaciones estándar
averages_e1 = {alg: np.mean(cost_e1_totals[alg]) for alg in cost_e1_totals}
averages_e2 = {alg: np.mean(cost_e2_totals[alg]) for alg in cost_e2_totals}
std_devs_e1 = {alg: np.std(cost_e1_totals[alg]) for alg in cost_e1_totals}
std_devs_e2 = {alg: np.std(cost_e2_totals[alg]) for alg in cost_e2_totals}

# Datos para los gráficos
algorithms = list(averages_e1.keys())
average_cost_e1 = [averages_e1[alg] for alg in algorithms]
average_cost_e2 = [averages_e2[alg] for alg in algorithms]
std_cost_e1 = [std_devs_e1[alg] for alg in algorithms]
std_cost_e2 = [std_devs_e2[alg] for alg in algorithms]

# Crear la figura y los ejes para los gráficos de costos
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Gráfico de barras para Cost E1 con barras de error
ax[0, 0].bar(algorithms, average_cost_e1, yerr=std_cost_e1, color='skyblue', capsize=5)
ax[0, 0].set_title('Costo promedio escenario 1')
ax[0, 0].set_xlabel('Algoritmo')
ax[0, 0].set_ylabel('Costo')

# Gráfico de barras para Cost E2 con barras de error
ax[0, 1].bar(algorithms, average_cost_e2, yerr=std_cost_e2, color='lightgreen', capsize=5)
ax[0, 1].set_title('Costo promedio escenario 2')
ax[0, 1].set_xlabel('Algoritmo')
ax[0, 1].set_ylabel('Costo')

# Gráfico de caja y bigotes para Cost E1
ax[1, 0].boxplot([cost_e1_totals[alg] for alg in algorithms], labels=algorithms)
ax[1, 0].set_title('Distribución de costos escenario 1')
ax[1, 0].set_xlabel('Algoritmo')
ax[1, 0].set_ylabel('Costo')

# Gráfico de caja y bigotes para Cost E2
ax[1, 1].boxplot([cost_e2_totals[alg] for alg in algorithms], labels=algorithms)
ax[1, 1].set_title('Distribución de costos escenario 2')
ax[1, 1].set_xlabel('Algoritmo')
ax[1, 1].set_ylabel('Costo')

# Ajustar el diseño y mostrar la gráfica de costos
plt.tight_layout()
plt.show()
