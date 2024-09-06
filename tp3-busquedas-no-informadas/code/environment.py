import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium import wrappers
import time
import random
import numpy as np

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
    
    def total_cost(self, path, scenario = 1):
        cost = 0
        for movement in path:
            if scenario == 1:
                cost += 1
            elif scenario == 2:
                cost += movement
        return cost

    def manhattan_distance(self, current_node, goal_node):
        x_position = current_node[0]
        y_position = current_node[1]
        x_distance = np.abs(x_position - goal_node[0])
        y_distance = np.abs(y_position - goal_node[1])
        return x_distance + y_distance
    
    def euclidean_distance(self, current_node, goal_node):
        x_position = current_node[0]
        y_position = current_node[1]
        x_distance = (x_position - goal_node[0]) ** 2
        y_distance = (y_position - goal_node[1]) ** 2
        return np.sqrt(x_distance + y_distance)





    