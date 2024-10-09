import csv
import hillClimbing
import simulatedAnnealing
import geneticAlgorithm
import environment

def save_data():
    hill_list = []
    annealing_list = []
    genetic_list = []
    env_sizes = [4, 8, 10]
    
    for i in range(len(env_sizes)):
        for _ in range(30):
            env = environment.generate_initial_solution(env_sizes[i])
            hill_list.append(("Hill climbing", i, hillClimbing.hill_climbing(env)))
            annealing_list.append(("Simulated annealing", i, simulatedAnnealing.simmulated_annealing(env, 0.99, 0.01)))
            genetic_list.append(("Genetic algorithm", i, geneticAlgorithm.geneticAlgorithm(env)))
    
    file_path = r'C:\Users\joaqu\OneDrive\Facultad\Tercer año\Sexto semestre\Inteligencia Artificial 1\ia-uncuyo-2024\tp5-busquedas-locales\results.csv'
    
    results = [
        hill_list,
        annealing_list,
        genetic_list
    ]
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["algorithm_name", "env_size", "solution", "solution_cost", "iterations", "time"])
        
        for result in results:
            for solution in result:
                writer.writerow(solution)
                
save_data()