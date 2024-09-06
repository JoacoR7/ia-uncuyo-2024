import random
import time

def find_path(env):
    space = env.space
    size = len(space)
    
    start = env.find_subject()
    goal = env.find_goal()
    
    movements = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    path = [start]
    path_movement = []
    current_position = start
    
    states_explored = 0
    start_time = time.time()
    
    while True:
        possible_moves = []
        # Realizo los movimientos disponibles desde la posición actual
        for i, move in enumerate(movements):
            next_row, next_column = current_position[0] + move[0], current_position[1] + move[1]
            if 0 <= next_row < size and 0 <= next_column < size:
                next_position = (next_row, next_column)
                if space[next_row][next_column] != 'H':
                    possible_moves.append(((next_row, next_column), i))
        
        if not possible_moves:
            path = path_movement = []
            break
        

        next_position, next_movement = random.choice(possible_moves)
        path.append(next_position)
        path_movement.append(next_movement)
        current_position = next_position
        states_explored += 1
        
        if current_position == goal:
            break
    
    total_time = time.time() - start_time
    return path, path_movement, states_explored, total_time