import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium import wrappers
import time
import random

class Environment:
    def __init__(self, size, agent_life, hole_rate=0, is_deterministic=True, scenario=1):
        self.size = size
        self.hole_rate = hole_rate
        self.agent_life = agent_life
        self.scenario = scenario  # 1 para costo fijo, 2 para costo variable
        self.space, self.initial_holes_count = self.generate_random_map_custom(size, hole_rate)
        self.env = gym.make('FrozenLake-v1', desc=self.space, render_mode='human', is_slippery=not is_deterministic).env
        self.env = wrappers.TimeLimit(self.env, agent_life)
        self.performance = 0

    def generate_random_map_custom(self, size, hole_rate=0):
        if hole_rate == 0:
            hole_rate = random.random()
        total_cells = size**2
        hole_cells_quantity = int(total_cells * hole_rate) - 2

        # Entorno inicializado con hielo
        space = [['F' for _ in range(size)] for _ in range(size)]
        
        # Posiciones del sujeto y del objetivo
        empty_cells = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(empty_cells)

        # Sujeto
        subject_position = empty_cells.pop()
        space[subject_position[0]][subject_position[1]] = "S"

        # Objetivo
        goal_position = empty_cells.pop()
        space[goal_position[0]][goal_position[1]] = "G"

        # Agujeros
        for _ in range(hole_cells_quantity):
            hole_position = empty_cells.pop()
            space[hole_position[0]][hole_position[1]] = "H"

        return space, hole_cells_quantity

    def render(self):
        self.env.render()
        time.sleep(5)

    def step(self, action):
        # Tomar la acción y obtener el siguiente estado, recompensa, y si el episodio ha terminado
        next_state, reward, done, info = self.env.step(action)
        
        # Costos asociados a las acciones según el escenario
        if self.scenario == 1:
            action_cost = 1  # Cada acción tiene un costo fijo de 1
        elif self.scenario == 2:
            action_cost = {
                0: 0,  # Moverse a la izquierda
                1: 1,  # Moverse hacia abajo
                2: 2,  # Moverse a la derecha
                3: 3   # Moverse hacia arriba
            }.get(action, 0)
        
        # Descontar el costo de la acción del rendimiento
        self.performance -= action_cost

        self.render()
        
        return next_state, reward, done, info

    def reset(self):
        self.performance = 0
        return self.env.reset()
    
    def find_subject(self):
        start = None
        for r in range(self.size):
            for c in range(self.size):
                if self.space[r][c] == 'S':
                    start = (r, c)
        return start
    
    def find_goal(self):
        goal = None
        for r in range(self.size):
            for c in range(self.size):
                if self.space[r][c] == 'G':
                    goal = (r, c)
        return goal



"""space = Environment(100, 5, 0.08)
env = space.env

state = env.reset()
print("Posición inicial del agente:", state[0])
done = truncated = False
i = 0
while not (done or truncated):
    i += 1
    print(i)
    time.sleep(1)
    action = env.action_space.sample() # Acción aleatoria
    next_state, reward, done, truncated, _ = env.step(action)
    print(f"Acción: {action}, Nuevo estado: {next_state}, Recompensa: {reward}")
    print(f"¿Ganó? (encontró el objetivo): {done}")
    print(f"¿Frenó? (alcanzó el máximo de pasos posible): {truncated}\n")
    state = next_state"""
    