import dataCollection
from collections import defaultdict
import matplotlib.pyplot as plt


data = dataCollection.load_data()

# Inicializar diccionarios para acumular los costos y contar las ocurrencias
cost_e1_totals = defaultdict(float)
cost_e2_totals = defaultdict(float)
states_totals = defaultdict(int)
counts = defaultdict(int)

# Acumular los valores por algoritmo
for row in data:
    algorithm = row['algorithm_name']
    cost_e1_totals[algorithm] += float(row['cost_e1'])
    cost_e2_totals[algorithm] += float(row['cost_e2'])
    states_totals[algorithm] += int(row['states_n'])
    counts[algorithm] += 1

# Calcular los promedios
averages = {}
for algorithm in counts:
    averages[algorithm] = {
        'average_cost_e1': cost_e1_totals[algorithm] / counts[algorithm],
        'average_cost_e2': cost_e2_totals[algorithm] / counts[algorithm],
        'average_states_n': states_totals[algorithm] / counts[algorithm],
    }

# Datos para el gráfico
algorithms = list(averages.keys())
average_cost_e1 = [averages[alg]['average_cost_e1'] for alg in algorithms]
average_cost_e2 = [averages[alg]['average_cost_e2'] for alg in algorithms]
average_states_n = [averages[alg]['average_states_n'] for alg in algorithms]

# Crear la figura y los ejes
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Gráfico de barras para Cost E1
ax[0].bar(algorithms, average_cost_e1, color='skyblue')
ax[0].set_title('Costo promedio Entorno 1')
ax[0].set_xlabel('Algoritmo')
ax[0].set_ylabel('Costo')

# Gráfico de barras para Cost E2
ax[1].bar(algorithms, average_cost_e2, color='lightgreen')
ax[1].set_title('Costo promedio Entorno 2')
ax[1].set_xlabel('Algoritmo')
ax[1].set_ylabel('Costo')

# Gráfico de barras para States N
ax[2].bar(algorithms, average_states_n, color='salmon')
ax[2].set_title('Promedio de estados visitados')
ax[2].set_xlabel('Algoritno')
ax[2].set_ylabel('Cantidad de estados')

# Ajustar el diseño y mostrar la gráfica
plt.tight_layout()
plt.show()