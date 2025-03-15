import gymnasium as gym
import numpy as np
import ale_py
import random

# Crear el entorno en modo RAM
gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-ram-v5", render_mode=None)

# Inicializar variables
N = 10000  # Número de pasos de juego
ram_history = []  # Lista para almacenar los valores de la RAM en cada paso
acciones_permitidas = [4, 5]

# Reiniciar el entorno
obs, _ = env.reset()
partidas = 0
# Jugar con acciones aleatorias y almacenar los valores de la RAM
for _ in range(N):
    action = acciones_permitidas[random.randint(0,1)]  # Acción aleatoria
    obs, _, done, _, _ = env.step(action)  # Ejecutar acción
    ram_history.append(obs.copy())  # Guardar la RAM actual
    if done:
        partidas += 1
        obs, _ = env.reset()  # Reiniciar si el juego termina

env.close()  # Cerrar el entorno

# Convertir la lista en un array de NumPy para analizar cambios
ram_history = np.array(ram_history)

# Contar cuántas veces cambia cada byte de la RAM
changes = np.sum(ram_history[:-1] != ram_history[1:], axis=0)

indices = []
j = 0
print(f"Se jugaron {partidas} partidas")
for i in changes:
    if i >= partidas * 3:
        indices.append(j)
    j += 1

print(f"Total de indices importantes {len(indices)}")
print(changes)
print(indices)
