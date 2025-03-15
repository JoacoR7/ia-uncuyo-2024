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
    
    # Crear diccionario con las métricas
    metricas = {
        "Métrica": ["Recompensa Promedio", "Recompensa Máxima", "Recompensa Mínima", "Pasos Promedio", "Tasa de Éxito"],
        "Valor": [recompensa_promedio, recompensa_maxima, recompensa_minima, pasos_promedio, tasa_exito]
    }
    return metricas

def graficar_recompensa_por_bloques(archivo_csv, intervalo=50, ruta_guardado=None, nombre_archivo="recompensa_por_bloques.png"):
    datos = pd.read_csv(archivo_csv)
    # Crear grupos de tamaño intervalo y calcular la media
    datos["Grupo"] = datos["Episodio"] // intervalo
    recompensa_promedio_por_bloque = datos.groupby("Grupo")["Recompensa Total"].mean()
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(recompensa_promedio_por_bloque.index * intervalo, recompensa_promedio_por_bloque, marker='o', linestyle='-', color="green", label=f"Promedio cada {intervalo} episodios")
    plt.xlim(RANGO_X_RECOMPENSA)
    plt.ylim(RANGO_Y_RECOMPENSA)
    plt.xlabel("Número de Episodio")
    plt.ylabel("Recompensa Promedio")
    plt.title(f"Recompensa Promedio cada {intervalo} Episodios")
    plt.legend()
    plt.grid()

    # Guardar imagen si se especifica
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico guardado en: {path_completo}")
    

def graficar_evolucion_pasos(archivo_csv, intervalo=50, ruta_guardado=None, nombre_archivo="evolucion_pasos_por_bloques.png"):
    datos = pd.read_csv(archivo_csv)
    # Crear grupos de tamaño intervalo y calcular la media de pasos
    datos["Grupo"] = datos["Episodio"] // intervalo
    pasos_promedio_por_bloque = datos.groupby("Grupo")["Pasos"].mean()
    
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(pasos_promedio_por_bloque.index * intervalo, pasos_promedio_por_bloque, marker='o', linestyle='-', color="blue", label=f"Promedio cada {intervalo} episodios")

    plt.xlim((0, 1000) )  # Rango fijo para el eje X
    plt.ylim(RANGO_Y_PASOS)  # Rango fijo para el eje Y
    plt.xlabel("Número de Episodio")
    plt.ylabel("Pasos Promedio")
    plt.title(f"Evolución de Pasos Promedio cada {intervalo} Episodios")
    plt.legend()
    plt.grid()

    # Guardar imagen si se especifica
    if ruta_guardado:
        path_completo = os.path.join(ruta_guardado, nombre_archivo)
        plt.savefig(path_completo, bbox_inches='tight')
        print(f"Gráfico de evolución de pasos guardado en: {path_completo}")
    

def guardar_en_zip(archivo_csv, nombre_zip="resultados.zip"):
    with zipfile.ZipFile(nombre_zip, "w") as zipf:
        # Graficar y agregar gráficos al ZIP
        graficar_pasos_vs_recompensa(archivo_csv, ruta_guardado=".")
        zipf.write("grafico_pasos_vs_recompensa.png", arcname="grafico_pasos_vs_recompensa.png")
        
        graficar_boxplots(archivo_csv, ruta_guardado=".")
        zipf.write("grafico_boxplot.png", arcname="grafico_boxplot.png")
        
        graficar_distribucion_recompensas(archivo_csv, ruta_guardado=".")
        zipf.write("distribucion_recompensas.png", arcname="distribucion_recompensas.png")
        
        graficar_evolucion_pasos(archivo_csv, ruta_guardado=".")
        zipf.write("evolucion_pasos_por_bloques.png", arcname="evolucion_pasos_por_bloques.png")
        
        graficar_recompensa_por_bloques(archivo_csv, ruta_guardado=".")
        zipf.write("recompensa_por_bloques.png", arcname="recompensa_por_bloques.png")
        
        # Calcular métricas y agregarlas al ZIP
        metricas = calcular_metricas_clave(archivo_csv)
        # Guardar las métricas en un archivo de texto
        with open("metricas_clave.txt", "w") as f:
            f.write("Métricas Clave:\n")
            for i in range(len(metricas["Métrica"])):
                f.write(f"- {metricas['Métrica'][i]}: {metricas['Valor'][i]:.2f}\n")
        
        zipf.write("metricas_clave.txt", arcname="metricas_clave.txt")

    print(f"Todos los archivos han sido guardados en el archivo ZIP: {nombre_zip}")


# Bucle para procesar archivos
for i in range(5000,35000, 5000):
    archivo = f"C:\\Users\\Victor\\Desktop\\Fuentes proyecto\\Ejercicios\\Prueba8\\resultados_ep{i}.csv"
    guardar_en_zip(archivo, f"resultados_ep{i}.zip")