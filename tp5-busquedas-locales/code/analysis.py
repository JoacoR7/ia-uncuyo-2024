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

plotear_resultados(datos_filtrados, 'Porcentaje Optimo')
plotear_resultados(datos_filtrados, 'Tiempo Promedio (s)')
plotear_resultados(datos_filtrados, 'Desviación Tiempo (s)')
plotear_resultados(datos_filtrados, 'Promedio Iteraciones')
plotear_resultados(datos_filtrados, 'Desviación Iteraciones')
