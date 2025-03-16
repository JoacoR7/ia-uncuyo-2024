import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# Definir rangos fijos para los ejes X e Y
RANGO_X_RECOMPENSA = (0, 1000)  # Rango fijo para el eje X en gráficos de recompensa
RANGO_Y_RECOMPENSA = (0, 1200)  # Rango fijo para el eje Y en gráficos de recompensa
RANGO_X_PASOS = (0, 2000)       # Rango fijo para el eje X en gráficos de pasos
RANGO_Y_PASOS = (0, 2000)       # Rango fijo para el eje Y en gráficos de pasos

def graficar_pasos_vs_recompensa(archivo_csv, ruta_guardado=None, nombre_archivo="grafico_pasos_vs_recompensa.png"):
    datos = pd.read_csv(archivo_csv)
    plt.figure(figsize=(10, 5))
    plt.scatter(datos['Pasos'], datos['Recompensa Total'], color="purple", alpha=0.5)
    plt.xlim(RANGO_X_PASOS)  # Rango fijo para el eje X
    plt.ylim(RANGO_Y_RECOMPENSA)  # Rango fijo para el eje Y
    plt.xlabel("Cantidad de Pasos")
    plt.ylabel("Recompensa Total")
    plt.title("Relación entre Cantidad de Pasos y Recompensa Obtenida")
    plt.grid()
    
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico guardado en: {path_completo}")

def graficar_boxplots(archivo_csv, ruta_guardado=None, nombre_archivo="grafico_boxplot.png"):
    datos = pd.read_csv(archivo_csv)
    plt.figure(figsize=(12, 5))
    plt.boxplot(datos['Recompensa Total'])
    plt.ylim(RANGO_Y_RECOMPENSA)  # Rango fijo para el eje Y
    plt.ylabel("Recompensa Total")
    plt.title("Distribución de la Recompensa")
    
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico guardado en: {path_completo}")

def graficar_distribucion_recompensas(archivo_csv, ruta_guardado=None, nombre_archivo="distribucion_recompensas.png"):
    datos = pd.read_csv(archivo_csv)
    plt.figure(figsize=(10, 5))
    plt.hist(datos['Recompensa Total'], bins=30, color='purple', edgecolor='black')
    plt.xlim(RANGO_X_RECOMPENSA)  # Rango fijo para el eje X
    plt.ylim(0, 500)  # Rango fijo para el eje Y (ajusta según tus datos)
    plt.title("Distribución de Frecuencia de Recompensas")
    plt.xlabel("Recompensa Total")
    plt.ylabel("Frecuencia")
    plt.grid()
    
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico de distribución de recompensas guardado en: {path_completo}")

def calcular_metricas_clave(archivo_csv):
    datos = pd.read_csv(archivo_csv)
    # Calcular métricas
    recompensa_promedio = datos['Recompensa Total'].mean()
    recompensa_maxima = datos['Recompensa Total'].max()
    recompensa_minima = datos['Recompensa Total'].min()
    pasos_promedio = datos['Pasos'].mean()
    tasa_exito = (datos['Recompensa Total'] >= 630).mean() * 100  # Ejemplo: éxito si recompensa >= 630
    
    # Calcular IQR y desviación estándar
    q1 = datos['Recompensa Total'].quantile(0.25)
    q3 = datos['Recompensa Total'].quantile(0.75)
    iqr = q3 - q1
    desviacion_estandar = datos['Recompensa Total'].std()

    # Crear diccionario con las métricas
    metricas = {
        "Métrica": ["Recompensa Promedio", "Recompensa Máxima", "Recompensa Mínima", "Pasos Promedio", "Tasa de Éxito", "IQR (Rango Intercuartílico)", "Desviación Estándar"],
        "Valor": [recompensa_promedio, recompensa_maxima, recompensa_minima, pasos_promedio, tasa_exito, iqr, desviacion_estandar]
    }
    return metricas
    
def graficar_evolucion_recompensa(lista_episodios, lista_promedio_recompensas, ruta_guardado=None, nombre_archivo="evolucion_recompensa.png"):
    # Crear el gráfico con una figura más ancha
    plt.figure(figsize=(15, 6))  # Aumenta el ancho de la figura
    plt.plot(lista_episodios, lista_promedio_recompensas, marker='o', linestyle='-', color='r')

    # Etiquetas y título
    plt.xlabel("Episodios (en bloques de 5000)")
    plt.ylabel("Recompensa Promedio")
    plt.title("Evolución de la Recompensa Promedio cada 5000 Episodios")
    plt.ylim((0, 500))  # Rango fijo para el eje Y

    # Añadir proyección de cada punto sobre el eje X con líneas punteadas
    for episodio, recompensa in zip(lista_episodios, lista_promedio_recompensas):
        plt.vlines(episodio, 0, recompensa, linestyles='dotted', colors='gray')

    # Añadir el valor del promedio arriba de cada punto
    for episodio, recompensa in zip(lista_episodios, lista_promedio_recompensas):
        plt.text(episodio, recompensa + 5, f'{recompensa:.2f}', ha='center', va='bottom', fontsize=9, color='black')

    # Ajustar las etiquetas del eje x para que estén más separadas
    plt.xticks(lista_episodios, rotation=45)  # Rota las etiquetas para mejor legibilidad

    # Guardar imagen si se especifica
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico de evolución de recompensas guardado en: {path_completo}")

def graficar_evolucion_winrate(lista_episodios, lista_winrate, ruta_guardado=None, nombre_archivo="evolucion_winrate.png"):
    # Crear el gráfico con una figura más ancha
    plt.figure(figsize=(15, 6))  # Aumenta el ancho de la figura
    plt.plot(lista_episodios, lista_winrate, marker='o', linestyle='-', color='r')

    # Etiquetas y título
    plt.xlabel("Episodios (en bloques de 5000)")
    plt.ylabel("Winrate")
    plt.title("Evolución del Winrate cada 5000 Episodios")
    plt.ylim((0, 2))  # Rango fijo para el eje Y

    # Añadir proyección de cada punto sobre el eje X con líneas punteadas
    for episodio, winrate in zip(lista_episodios, lista_winrate):
        plt.vlines(episodio, 0, winrate, linestyles='dotted', colors='gray')

    # Añadir el valor del promedio arriba de cada punto
    for episodio, winrate in zip(lista_episodios, lista_winrate):
        plt.text(episodio, winrate + 0.015, f'{winrate:.2f}', ha='center', va='bottom', fontsize=9, color='black')

    # Ajustar las etiquetas del eje x para que estén más separadas
    plt.xticks(lista_episodios, rotation=45)  # Rota las etiquetas para mejor legibilidad

    # Guardar imagen si se especifica
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico de evolución de winrate guardado en: {path_completo}")

def graficar_evolucion_pasos(lista_episodios, lista_promedio_pasos, ruta_guardado=None, nombre_archivo="evolucion_pasos.png"):
    # Crear el gráfico con una figura más ancha
    plt.figure(figsize=(15, 6))  # Aumenta el ancho de la figura
    plt.plot(lista_episodios, lista_promedio_pasos, marker='o', linestyle='-', color='r')

    # Etiquetas y título
    plt.xlabel("Episodios (en bloques de 5000)")
    plt.ylabel("Pasos Promedio")
    plt.title("Evolución de la Cantidad de Pasos Promedio cada 5000 Episodios")
    plt.ylim((0, 1500))  # Rango fijo para el eje Y

    # Añadir proyección de cada punto sobre el eje X con líneas punteadas
    for episodio, pasos in zip(lista_episodios, lista_promedio_pasos):
        plt.vlines(episodio, 0, pasos, linestyles='dotted', colors='gray')

    # Añadir el valor del promedio arriba de cada punto
    for episodio, pasos in zip(lista_episodios, lista_promedio_pasos):
        plt.text(episodio - 15, pasos + 10, f'{pasos:.2f}', ha='center', va='bottom', fontsize=7, color='black')

    # Ajustar las etiquetas del eje x para que estén más separadas
    plt.xticks(lista_episodios, rotation=45)  # Rota las etiquetas para mejor legibilidad

    # Guardar imagen si se especifica
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico de evolución de pasos promedio guardado en: {path_completo}")
 

def guardar_en_zip(archivo_csv, nombre_zip="resultados.zip"):

    with zipfile.ZipFile(nombre_zip, "w") as zipf:
        # Graficar y agregar gráficos al ZIP
        graficar_pasos_vs_recompensa(archivo_csv, ruta_guardado=".")
        zipf.write("grafico_pasos_vs_recompensa.png", arcname="grafico_pasos_vs_recompensa.png")
        
        graficar_boxplots(archivo_csv, ruta_guardado=".")
        zipf.write("grafico_boxplot.png", arcname="grafico_boxplot.png")
        
        graficar_distribucion_recompensas(archivo_csv, ruta_guardado=".")
        zipf.write("distribucion_recompensas.png", arcname="distribucion_recompensas.png")
                
        # Calcular métricas y agregarlas al ZIP
        metricas = calcular_metricas_clave(archivo_csv)
        # Guardar las métricas en un archivo de texto
        with open("metricas_clave.txt", "w") as f:
            f.write("Métricas Clave:\n")
            for i in range(len(metricas["Métrica"])):
                f.write(f"- {metricas['Métrica'][i]}: {metricas['Valor'][i]:.2f}\n")
        
        zipf.write("metricas_clave.txt", arcname="metricas_clave.txt")

    print(f"Todos los archivos han sido guardados en el archivo ZIP: {nombre_zip}")
    return metricas


lista_recompensas_promedio = []
lista_pasos_promedio = []
lista_episodios = []
lista_winrate = []
# Bucle para procesar archivos
for i in range(5000,135000,5000):
    archivo = f"C:\\Users\\Victor\\Desktop\\Fuentes proyecto\\Ejercicios\\Prueba8\\resultados_ep{i}.csv"
    metricas = guardar_en_zip(archivo, f"resultados_ep{i}.zip")
    lista_recompensas_promedio.append(float(metricas["Valor"][0]))
    lista_pasos_promedio.append(float(metricas["Valor"][3]))
    lista_winrate.append(float(metricas["Valor"][4]))
    lista_episodios.append(i)


graficar_evolucion_recompensa(lista_episodios, lista_recompensas_promedio, ruta_guardado=".", nombre_archivo="evolucion_recompensa.png")
graficar_evolucion_winrate(lista_episodios, lista_winrate, ruta_guardado=".", nombre_archivo="evolucion_winrate.png")
graficar_evolucion_pasos(lista_episodios, lista_pasos_promedio, ruta_guardado=".", nombre_archivo="evolucion_pasos.png")