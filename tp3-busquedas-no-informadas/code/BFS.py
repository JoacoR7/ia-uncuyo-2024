from collections import deque

def find_path(env):
    space = env.space
    size = len(space)
    
    # Busco las posiciones del sujeto y del objetivo
    start = env.find_subject()
    goal = env.find_goal()
    
    # Movimientos: izquierda, abajo, derecha, arriba
    movements = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    # Inicializo cola con la posición del sujeto
    queue = deque([start])
    # Diccionario para rastrear el camino, incluyendo el movimiento realizado
    parent = {start: (None, None)}  # (posición previa, movimiento)
    
    while queue:
        current_node = queue.popleft()
        
        if current_node == goal:
            path = []
            while current_node:
                prev, move = parent[current_node]
                if move is not None:
                    path.append(move)
                current_node = prev
            path.reverse()
            return path
        
        for i, move in enumerate(movements):
            next_row, next_column = current_node[0] + move[0], current_node[1] + move[1]
            # ¿Está dentro del espacio?
            if 0 <= next_row < size and 0 <= next_column < size: 
                next_pos = (next_row, next_column)
                # ¿Se mueve a una posición sólida?
                if space[next_row][next_column] != 'H' and next_pos not in parent:
                    queue.append(next_pos)
                    parent[next_pos] = (current_node, i) 
    
    return None
