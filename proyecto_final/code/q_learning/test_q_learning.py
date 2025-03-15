import gymnasium as gym
import numpy as np
import pickle
import ale_py.roms
import ale_py
import os
import glob
import csv

# Ruta donde están almacenadas las Q-Tables
ruta_q_tables = r"C:\Users\Victor\Desktop\Fuentes proyecto\Ejercicios\Prueba8"

# Registrar el entorno
gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode=None, obs_type="ram")

# Definir acciones permitidas
acciones_permitidas = [4, 5]  # Disparar, moverse izquierda y disparar, moverse derecha y disparar

# Índices relevantes en la RAM
indices_relevantes = [2, 3, 4, 6, 8, 9, 10, 11, 17, 18, 19, 20, 21, 26, 28, 30, 42, 43, 44,
                      45, 46, 47, 48, 49, 50, 51, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                      82, 83, 84, 85, 87, 88, 90, 104, 106, 109, 110, 111, 114, 118, 120, 121,
                      124, 125, 126, 127]

# Función para encontrar los archivos Q-table y ordenarlos por número de episodios
def obtener_lista_q_tables():
    archivos = glob.glob(os.path.join(ruta_q_tables, "tabla_Q_ep*.pkl"))
    
    # Extraer los números de episodios y ordenar
    archivos_ordenados = sorted(archivos, key=lambda x: int(x.split("_ep")[1].split(".pkl")[0]))
    
    return archivos_ordenados

# Cargar la tabla Q desde un archivo
def cargar_tabla_Q(nombre_archivo):
    try:
        with open(nombre_archivo, "rb") as archivo:
            return pickle.load(archivo)
    except Exception as e:
        print(f"Error al cargar la tabla Q ({nombre_archivo}): {e}")
        exit()

# Discretizar la RAM
def discretizar(valor):
    return valor // (256 // 2)

# Obtener estado discretizado
def obtener_estado_discretizado(ram):
    estado_relevante = [ram[i] for i in indices_relevantes]
    return tuple(discretizar(estado) for estado in estado_relevante)

# Simular partidas y guardar resultados
def simular_partidas(env, Q, num_partidas=1000, archivo_csv="resultados.csv"):
    with open(archivo_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episodio", "Recompensa Total", "Pasos"])
        
        for episodio in range(num_partidas):
            estado, _ = env.reset()  # Semilla única
            estado_discretizado = obtener_estado_discretizado(estado)
            
            done = False
            total_recompensa = 0
            pasos = 0
            
            while not done:
                if estado_discretizado in Q:
                    accion_idx = np.argmax(Q[estado_discretizado])
                else:
                    accion_idx = 0  # Acción por defecto (ej: disparar)
                
                accion_real = acciones_permitidas[accion_idx]
                siguiente_estado, recompensa, done, _, _ = env.step(accion_real)
                estado_discretizado = obtener_estado_discretizado(siguiente_estado)
                
                total_recompensa += recompensa
                pasos += 1
                
            writer.writerow([episodio + 1, total_recompensa, pasos])
            print(f"Episodio {episodio + 1}: Recompensa {total_recompensa}, Pasos {pasos}")
    
    print(f"Simulación completada. Resultados guardados en {archivo_csv}")

# Ejecutar el código
archivos_q_tables = obtener_lista_q_tables()

if not archivos_q_tables:
    print("No se encontraron archivos de Q-table en la ruta especificada.")
    exit()

for archivo_q in archivos_q_tables:
    num_episodios = int(archivo_q.split("_ep")[1].split(".pkl")[0])  # Extraer el número de episodios
    Q = cargar_tabla_Q(archivo_q)
    simular_partidas(env, Q, num_partidas=1000, archivo_csv=f"resultados_ep{num_episodios}.csv")
    Q = None

env.close()
