import csv
import numpy as np
import random
import pickle  


import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode=None, obs_type="ram")
# Definir acciones permitidas
acciones_permitidas = [4, 5]  
n_actions = len(acciones_permitidas)  

# Índices relevantes en la RAM
indices_relevantes = [2, 3, 4, 6, 8, 9, 10, 11, 17, 18,
                      19, 20, 21, 26, 28, 30, 42, 43, 44,
                      45, 46, 47, 48, 49, 50, 51, 70, 72,
                      73, 74, 75, 76, 77, 78, 79, 80, 81,
                      82, 83, 84, 85, 87, 88, 90, 104, 106,
                      109, 110, 111, 114, 118, 120, 121, 124, 125, 126, 127]

# Cargar la tabla Q desde el archivo
def cargar_tabla_Q(nombre_archivo):
    try:
        with open(nombre_archivo, "rb") as archivo:
            return pickle.load(archivo)
    except Exception as e:
        print(f"Error al cargar la tabla Q: {e}")
        return {}

# Guardar la tabla Q
def guardar_tabla_Q(nombre_archivo="tabla_Q.pkl"):
    try:
        with open(nombre_archivo, "wb") as archivo:
            pickle.dump(Q, archivo, protocol=4)
            print(f"Tabla Q guardada en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar la tabla Q: {e}")

# Inicializar CSV para almacenar epsilon
def inicializar_csv(nombre_csv):
    with open(nombre_csv, mode='w', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["Episodio", "Epsilon"])

# Guardar epsilon en el CSV
def guardar_epsilon_csv(nombre_csv, episodio, epsilon):
    with open(nombre_csv, mode='a', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([episodio, epsilon])

# Diccionario Q (se carga desde el archivo)
Q = cargar_tabla_Q("/kaggle/input/tablaqv2/tabla_Q_ep10000.pkl")

# Función para discretizar un valor de la RAM (0-255)
def discretizar(valor):
    return valor // (256 // 2)  

# Convertir los bytes relevantes en un estado discretizado
def obtener_estado_discretizado(ram):
    estado_relevante = [ram[i] for i in indices_relevantes]
    estado_discretizado = tuple(discretizar(estado) for estado in estado_relevante)
    return estado_discretizado

# Inicializar la tabla Q si el estado no existe
def inicializar_estado(estado):
    if estado not in Q:
        Q[estado] = np.random.uniform(low=-0.01, high=0.01, size=n_actions)  

# Actualizar la tabla Q
def actualizar_Q(estado, accion_idx, recompensa, siguiente_estado, alpha, gamma):
    inicializar_estado(estado)
    inicializar_estado(siguiente_estado)
    max_q_siguiente = np.max(Q[siguiente_estado])
    Q[estado][accion_idx] += alpha * (recompensa + gamma * max_q_siguiente - Q[estado][accion_idx])

# Selección de acción con ε-greedy
def seleccionar_accion(estado, epsilon):
    inicializar_estado(estado)
    if random.random() < epsilon:  
        return random.randint(0, n_actions - 1)
    else:  
        return np.argmax(Q[estado])

def jugar_partida(env, episodios_totales, 
                  epsilon, epsilon_min, epsilon_max, epsilon_incremento, 
                  alpha_initial, gamma, nombre_csv):
    
    alpha = alpha_initial  # Tasa de aprendizaje inicial
    inicializar_csv(nombre_csv)
    epsilon_aumentando = False  # Flag para controlar la dirección del cambio de epsilon

    for episodio in range(episodios_totales):
        estado, _ = env.reset()
        estado_discretizado = obtener_estado_discretizado(estado)

        done = False
        total_recompensa = 0

        while not done:
            accion_idx = seleccionar_accion(estado_discretizado, epsilon)
            accion_real = acciones_permitidas[accion_idx]
            siguiente_estado, recompensa, done, _, _ = env.step(accion_real)
            siguiente_estado_discretizado = obtener_estado_discretizado(siguiente_estado)
            actualizar_Q(estado_discretizado, accion_idx, recompensa, siguiente_estado_discretizado, alpha, gamma)
            estado_discretizado = siguiente_estado_discretizado
            total_recompensa += recompensa

        # Aplicar Learning Rate Decay
        alpha = max(alpha_initial * (0.9995 ** episodio), 0.1)

        print(f"Episodio {episodio + 1}: Recompensa total = {total_recompensa}, Epsilon = {epsilon:.4f}, Alpha = {alpha:.4f}")

        # Guardar la tabla Q cada 5000 episodios
        if (episodio + 1) % 5000 == 0:
            nombre_archivo = f"tabla_Q_ep{episodio + 1}.pkl"
            guardar_tabla_Q(nombre_archivo)

        # Guardar epsilon en el CSV
        guardar_epsilon_csv(nombre_csv, episodio + 1, epsilon)

        # Control de epsilon (cíclico entre 0.1 y 0.5)
        if epsilon_aumentando:
            epsilon += epsilon_incremento
            if epsilon >= epsilon_max:
                epsilon = epsilon_max
                epsilon_aumentando = False  # Empezar a decrecer
        else:
            epsilon -= epsilon_incremento
            if epsilon <= epsilon_min:
                epsilon = epsilon_min
                epsilon_aumentando = True  # Empezar a aumentar

# Parámetros de entrenamiento
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 0.5
epsilon_incremento = 0.001  # Ajustar la velocidad del cambio de epsilon
alpha_initial = 0.5  # Tasa de aprendizaje inicial
gamma = 0.95  
episodios_totales = 40000  
nombre_csv = "epsilon_values.csv"

jugar_partida(env, episodios_totales, epsilon, epsilon_min, epsilon_max, epsilon_incremento, 
              alpha_initial, gamma, nombre_csv)

# Guardar la tabla Q final
guardar_tabla_Q("tabla_Q_final.pkl")
env.close()
