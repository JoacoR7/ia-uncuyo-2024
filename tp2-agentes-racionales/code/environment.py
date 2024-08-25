import random
import numpy as np

class Environment:
    def __init__(self, sizeX, sizeY, dirt_rate):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.agent_pos = (random.randint(0, self.sizeX-1), random.randint(0, self.sizeY-1))
        self.agent_act = 0 #Contador de acciones del agente (movimientos, limpieza, idle)
        self.agent_performance_action_count = 0
        self.space, self.initial_dirty_cells_count = self._generate_space(dirt_rate)
        self.performance = 0
        self.dirt_rate = dirt_rate
        self.action_count = [] 

    def _generate_space(self, dirt_rate):
        # Clcula el número total de celdas y el número de celdas sucias
        total_cells = self.sizeX * self.sizeY
        num_dirty_cells = int(total_cells * dirt_rate)
        
        #Crea una lista de celdas, donde 1 representa una celda sucia y 0 una limpia
        cells = [1] * num_dirty_cells + [0] * (total_cells - num_dirty_cells)
        
        random.shuffle(cells)
        space = np.array(cells).reshape((self.sizeX, self.sizeY))
        return space, num_dirty_cells

    def accept_action(self, action):
        #Actualiza la posición del agente y limpia si es necesario
        x, y = self.agent_pos
        self.agent_act += 1

        if action == 'Up' and x > 0:
            x -= 1
        elif action == 'Down' and x < self.sizeX - 1:
            x += 1
        elif action == 'Left' and y > 0:
            y -= 1
        elif action == 'Right' and y < self.sizeY - 1:
            y += 1
        elif action == 'Clean':
            if self.space[x,y]:  #Si la celda está sucia
                self.space[x,y] = 0
                self.performance += 1
                self.agent_performance_action_count = self.agent_act
                self.action_count.append(self.agent_act)

        elif action == 'Idle':
            pass

        self.agent_pos = (x, y)

    def is_dirty(self):
        #Devuelve True si la celda actual está sucia
        i, j = self.agent_pos
        return int(self.space[i,j])

    def get_performance(self):
        #Devuelve la medida de rendimiento
        return self.performance

    def print_environment(self):
        #Imprime el estado actual del entorno, incluyendo la posición del agente
        for i in range(self.sizeX):
            for j in range(self.sizeY ):
                if (i, j) == self.agent_pos:
                    if self.space[i,j]:
                        print("@", end=" ")  #Posición del agente con ubicación sucia
                    else:
                        print("A", end=" ") #Posición del agente con ubicación sucia
                elif self.space[i,j]:
                    print("D", end=" ")  #Celda sucia
                else:
                    print("C", end=" ")  #Celda limpia
            print()
        print(f"Performance: {self.performance}")
    
    def agent_action(self, agent, quantity):
        for i in range(quantity):
            agent.think()
        return self.performance, self.agent_performance_action_count

def max_performance_points(environments):
    max_performance = 0
    best_env = None

    for env in environments:
        performance = env.get_performance()
        if performance > max_performance:
            max_performance = performance
            best_env = env

    return best_env, max_performance
