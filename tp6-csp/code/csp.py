import environment
from collections import deque
import copy

env = environment.generate_environment(8)

def backtrack(env, column, N):
    if(column == N):
        return True
    
    for row in range(N):
        env[column] = row
        if(environment.objective_function(env) == 0):
            if backtrack(env, column+1, N):
                return True
            env[column] = -1
            env[column+1] = -1

def N_queens_backtrack(env):
    backtrack(env, 1, len(env))
    return env

def forward_check(env, column, domain_list):
    N = len(env)
    for i in range(column+1, N):
        # Elimino la fila actual de la siguiente columna
        if env[column] in domain_list[i]:
            domain_list[i].remove(env[column])
            
        # Elimino la diagonal abajo
        diagonal_abajo = env[column] + (i - column)
        if diagonal_abajo < N and diagonal_abajo in domain_list[i]:
            domain_list[i].remove(diagonal_abajo)
        
        # Elimino la diagonal arriba 
        diagonal_arriba = env[column] - (i - column)
        if diagonal_arriba >= 0 and diagonal_arriba in domain_list[i]:
            domain_list[i].remove(diagonal_arriba)

        

def forward(env, domain_list, column):
    N = len(env)
    if column == N:
        return env  
    
    for row in domain_list[column]:
        env[column] = row
        
        if environment.objective_function(env) == 0:
            new_domain_list = copy.deepcopy(domain_list)
            
            # Aplicar forward checking
            forward_check(env, column, new_domain_list)
            
            # Comprobar si forward checking dejó dominios válidos
            if all(new_domain_list[c] for c in range(column+1, N)):
                result = forward(env, new_domain_list, column + 1)
                if result is not None:
                    return result  # Solución encontrada
    
    # Si ninguna opción funcionó, hacemos backtracking
    env[column] = -1
    return None

def N_queens_forward(env):
    N = len(env)
    domain_list = [[i for i in range(N)] for _ in range(N)]
    forward_check(env, 0, domain_list)
    return forward(env, domain_list, 1)
    
print(env)
print(N_queens_forward(env))