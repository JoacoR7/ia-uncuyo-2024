import csv

def initialize_csv(filename = r'C:\Users\joaqu\OneDrive\Facultad\Tercer año\Sexto semestre\Inteligencia Artificial 1\ia-uncuyo-2024\tp5-busquedas-locales\tp5-Nreinas.csv'):
    # Crea un nuevo archivo con encabezados
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Algorithm Name', 'Solution', 'Env Size', 'Objective Function', 'Iterations', 'Time'])

def save_results(algorithm_name, objective_function_value, iterations, elapsed_time, solution, env_size, filename = r'C:\Users\joaqu\OneDrive\Facultad\Tercer año\Sexto semestre\Inteligencia Artificial 1\ia-uncuyo-2024\tp5-busquedas-locales\tp5-Nreinas.csv'):
    # Abre el archivo en modo append
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Escribe una fila con los resultados
        writer.writerow([algorithm_name, env_size, objective_function_value, iterations, elapsed_time, solution])
