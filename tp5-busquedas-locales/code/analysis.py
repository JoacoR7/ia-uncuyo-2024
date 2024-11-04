import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


filename = r'C:\Users\joaqu\OneDrive\Facultad\Tercer año\Sexto semestre\Inteligencia Artificial 1\ia-uncuyo-2024\tp5-busquedas-locales\tp5-Nreinas.csv'
df = pd.read_csv(filename)

# Definir la condición de "óptimo"
estado_optimo = 0  # Ajusta según sea necesario
df['Es Optimo'] = df['Objective Function'] == estado_optimo

# Agrupar datos por 'Algorithm Name' y 'Env Size' con las métricas solicitadas
resultados = df.groupby(['Algorithm Name', 'Env Size']).apply(lambda group: pd.Series({
    'Porcentaje Optimo': group['Es Optimo'].mean() * 100,
    'Tiempo Promedio (s)': group.loc[group['Es Optimo'], 'Time'].mean(),
    'Desviación Tiempo (s)': group.loc[group['Es Optimo'], 'Time'].std(),
    'Promedio Iteraciones': group.loc[group['Es Optimo'], 'Iterations'].mean(),
    'Desviación Iteraciones': group.loc[group['Es Optimo'], 'Iterations'].std()
})).reset_index()

# Función para filtrar datos
def filtrar_datos(algoritmo=None, env_size=None):
    filtrado = resultados.copy()
    if algoritmo:
        filtrado = filtrado[filtrado['Algorithm Name'] == algoritmo]
    if env_size:
        filtrado = filtrado[filtrado['Env Size'] == env_size]
    return filtrado

# Función para plotear datos
def plotear_resultados(data, metric):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Env Size', y=metric, hue='Algorithm Name', data=data)
    plt.title(f"{metric} por Tamaño de Entorno y Algoritmo")
    plt.ylabel(metric)
    plt.xlabel("Tamaño del tablero")
    plt.legend(title='Algorithm Name')
    plt.show()

# Ejemplo de uso: Filtrar por algoritmo o tamaño de entorno
algoritmo = 'hill_climbing'  # Cambia el nombre del algoritmo si deseas filtrar por uno específico, o pon None
env_size = 4  # Cambia el tamaño de entorno si deseas filtrar por uno específico, o pon None

# Filtrar los datos
datos_filtrados = filtrar_datos(algoritmo=None, env_size=None)

#plotear_resultados(datos_filtrados, 'Porcentaje Optimo')
#plotear_resultados(datos_filtrados, 'Tiempo Promedio (s)')
#plotear_resultados(datos_filtrados, 'Desviación Tiempo (s)')
#plotear_resultados(datos_filtrados, 'Promedio Iteraciones')
#plotear_resultados(datos_filtrados, 'Desviación Iteraciones')

def plotear_iteraciones(objective_function_results):
    # Nombres de los algoritmos para etiquetar cada gráfico
    algorithms = ["Hill Climbing", "Simulated Annealing", "Genetic Algorithm"]

    # Crear subplots: 1 fila, 3 columnas
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Generar un gráfico independiente para cada algoritmo
    for i, results in enumerate(objective_function_results):
        axs[i].plot(results, label=f"{algorithms[i]} Fitness")
        axs[i].set_xlabel("Iteración")
        axs[i].set_ylabel("Valor de Fitness")
        axs[i].set_title(f"{algorithms[i]}")
        axs[i].legend()
        axs[i].grid(True)

    # Ajustar el layout
    plt.suptitle("Comparación de Algoritmos por Valor de Fitness en cada Iteración")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar para que el título no se superponga
    plt.show()
