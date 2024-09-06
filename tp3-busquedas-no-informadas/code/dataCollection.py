import csv
import BFS
import DFS
import UCS
import environment
import randomAgent
import ASearch


def save_data():
    BFS_list = []
    DFS_list = []
    DFS_limited_list = []
    UCS_list = []
    ASearch_list = []

    def append_solution(solution, list, env_number, env):
        if solution[0] == None:
            list.append([None, env_number, solution[2], 0, 0, solution[3], False])
        else:
            list.append([None, env_number, solution[2], len(solution[1]), env.total_cost(solution[1], scenario=2), solution[3], True])


    for i in range(30):
        env = environment.Environment(size=100, agent_life=1000, hole_rate=0.08)
        append_solution(BFS.find_path(env), BFS_list, i, env)
        append_solution(DFS.find_path(env), DFS_list, i, env)
        append_solution(DFS.find_path(env, depth_limit=10), DFS_limited_list, i, env)
        append_solution(UCS.find_path(env), UCS_list, i, env)
        append_solution(ASearch.find_path(env), ASearch_list, i, env)
        
        env2 = env
        env2.scenario = 2
        solution = UCS.find_path(env)
        UCS_list[i][4] = env.total_cost(solution[1], scenario=2)
        solution = ASearch.find_path(env)
        ASearch_list[i][4] = env.total_cost(solution[1], scenario=2)
        
        BFS_list[i][0] = "BFS"
        DFS_list[i][0] = "DFS"
        DFS_limited_list[i][0] = "DFSL"
        UCS_list[i][0] = "UCS"
        ASearch_list[i][0] = "A*"

    # Datos de ejemplo, reemplaza con los datos reales
    results = [
        BFS_list,
        DFS_list,
        DFS_limited_list,
        UCS_list,
        ASearch_list
    ]

    # Ruta del archivo CSV
    file_path = r'C:\Users\joaqu\Desktop\no-informada-results.csv'

    # Escribir los resultados en el archivo CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["algorithm_name", "env_n", "states_n", "cost_e1", "cost_e2", "time", "solution_found"])
        
        
        # Escribir las filas de datos
        for result in results:
            for solution in result:
                writer.writerow(solution)

def load_data():
    file_path = r'C:\Users\joaqu\Desktop\no-informada-results.csv'
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        data = [row for row in reader]
        
    return data
