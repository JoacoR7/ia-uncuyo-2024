import math
import time

def find_path(env, depth_limit=math.inf):
    space = env.space
    size = len(space)
    
    # Busco las posiciones del sujeto y del objetivo
    start = env.find_subject()
    goal = env.find_goal()
    
    # Movimientos: izquierda, abajo, derecha, arriba
    movements = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    # Inicializo la pila con la posición del sujeto y la profundidad inicial de 0
    stack = [(start, 0)]
    # Camino y movimiento
    parent = {start: (None, None)}  # (posición previa, movimiento)
    
    states_explored = 0
    start_time = time.time()  

    while stack:
        current_node, depth = stack.pop()
        
        if current_node == goal:
            path_movement = []
            path = []
            while current_node:
                parent_node, move = parent[current_node]
                path.append(current_node)
                if move is not None:
                    path_movement.append(move)
                current_node = parent_node
            path.reverse()
            total_time = time.time() - start_time  

            if len(path) > env.agent_life:
                return None, None, 0, total_time 
            
            return path, path_movement, states_explored, total_time 
        
        # Solo continuar si no se alcanzó el límite de profundidad
        if depth < depth_limit:
            for i, move in enumerate(movements):
                next_row, next_column = current_node[0] + move[0], current_node[1] + move[1]
                # Verifico que la siguiente posición esté dentro de los límites del espacio
                if 0 <= next_row < size and 0 <= next_column < size: 
                    next_pos = (next_row, next_column)
                    # Verifico que la siguiente posición no sea un obstáculo y no se haya visitado
                    if space[next_row][next_column] != 'H' and next_pos not in parent:
                        states_explored += 1
                        stack.append((next_pos, depth + 1))
                        parent[next_pos] = (current_node, i) 
    
    total_time = time.time() - start_time
    return None, None, 0, total_time  # states_explored = 0
