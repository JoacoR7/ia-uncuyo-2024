import environment
import time
                
def generate_neighbors(solution):
    neighbors = []
    n = len(solution)
    for row in range(n):
        for column in range(n):
            if(row != solution[column]):
                neighbor = solution.copy()
                neighbor[column] = row
                neighbors.append(neighbor)    
    return neighbors

def best_neighbor(solution):
    neighbors = generate_neighbors(solution)
    best_solution_cost = environment.objective_function(solution)
    best_solution = solution
    for neighbor in neighbors:
        cost = environment.objective_function(neighbor)
        if (cost < best_solution_cost):
            best_solution = neighbor
            best_solution_cost = cost
    return best_solution, best_solution_cost
    
def hill_climbing(solution):
    start = time.time()
    actual_solution = solution
    actual_cost = environment.objective_function(actual_solution)
    iterations = 1
    while True and actual_cost != 0:
        new_solution, new_cost = best_neighbor(actual_solution)
        
        if(new_cost >= actual_cost):
            break
        
        actual_solution = new_solution
        actual_cost = new_cost
        
        if(actual_cost == 0):
            break
        iterations += 1
    
    return actual_solution, actual_cost, iterations, time.time() - start