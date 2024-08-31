import heapq

def find_path(env, is_variable=False):
    space = env.space
    size = len(space)
    
    # Movimientos y sus direcciones codificadas
    movements = [((0, -1), 0), ((1, 0), 1), ((0, 1), 2), ((-1, 0), 3)]
    
    
    if is_variable:
        movement_costs = [0, 1, 2, 3]
    else:
        movement_costs = [1, 1, 1, 1]
    
    # Busco las posiciones del sujeto y del objetivo
    start = env.find_subject()
    goal = env.find_goal()
    
    # Inicializo la cola de prioridad con la posición del sujeto y costo inicial de 0
    priority_queue = [(0, start)]  # (costo acumulado, posición actual)
    # Diccionario para rastrear el camino
    parent = {start: (None, None)}  # (posición previa, movimiento)
    # Diccionario para rastrear el costo mínimo para llegar a cada nodo
    cost = {start: 0}
    
    while priority_queue:
        current_node_cost, current_node = heapq.heappop(priority_queue)
        
        if current_node == goal:
            path = []
            while current_node:
                prev, move = parent[current_node]
                if move is not None:
                    path.append(move)
                current_node = prev
            path.reverse()
            return path
        
        for (move, direction) in movements:
            next_row, next_column = current_node[0] + move[0], current_node[1] + move[1]
            # Verificar que la siguiente posición esté dentro de los límites del espacio
            if 0 <= next_row < size and 0 <= next_column < size: 
                next_pos = (next_row, next_column)
                if space[next_row][next_column] != 'H':
                    next_columncost = current_node_cost + movement_costs[direction] 
                    # Si el camino que encuentro es menos costoso, lo añado
                    if next_pos not in cost or next_columncost < cost[next_pos]:
                        cost[next_pos] = next_columncost
                        priority_queue.append((next_columncost, next_pos))
                        heapq.heapify(priority_queue)  # Mantener la cola de prioridad ordenada
                        parent[next_pos] = (current_node, direction)
    
    return None
