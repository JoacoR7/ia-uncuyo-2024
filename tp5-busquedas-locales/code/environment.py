import random

def generate_initial_solution(N):
    return [random.randint(0, N-1) for _ in range(N)]

def objective_function(solution):
    count = 0
    for i in range(len(solution)):
        count += check_diagonals(i, solution) + check_horizontal(i, solution)
    return count

def check_diagonals(column, solution):
    count = 0
    row = solution[column]
    N = len(solution)
    # Diagonal superior
    if(column < N - 1 and row > 0):
        for i in range(1, N):
            if(row - i >= 0 and column + i < N):
                if(solution[column+i] == row - i):
                    count += 1
                    break
            else:
                break
    # Diagonal inferior
    if(column < N - 1 and row < N - 1):
        for i in range(1, N):
            if(row - 1 < N and column + i < N):
                if(solution[column+i] == row + i):
                    count += 1
                    break
            else:
                break
    return count

def check_horizontal(column, solution):
    count = 0
    row = solution[column]
    N = len(solution)
    # Derecha
    if(column < N - 1):
        for i in range(1, N):
            if(column + i <= N-1):
                if(solution[column+i] == row):
                    count += 1
                    break
            else:
                break
    return count