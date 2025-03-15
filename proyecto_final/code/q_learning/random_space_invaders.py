import ale_py.roms
import gymnasium as gym
import ale_py
import random
import csv

# Configuración del entorno
gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode=None, obs_type="ram")

# Acciones permitidas
actions = [4, 5]

# Nombre del archivo CSV
csv_filename = "random_simulation_results.csv"

# Inicializar CSV con encabezado
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episodio", "Recompensa Total", "Pasos"])

# Simulación de 1000 episodios
num_episodios = 1000

for episodio in range(1, num_episodios + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0  # Contador de pasos en el episodio

    while not done:
        action = random.choice(actions)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state

    # Guardar datos del episodio en el CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episodio, total_reward, steps])

    print(f"Episodio {episodio}: Recompensa Total = {total_reward}, Pasos = {steps}")

# Cerrar el entorno
env.close()

print(f"Simulación completada. Resultados guardados en '{csv_filename}'.")
