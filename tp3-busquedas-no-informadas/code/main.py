import BFS
import DFS
import UCS
import environment
import randomAgent
import ASearch

BFS_list = []
DFS_list = []
DFS_limited_list = []
UCS_list = []
ASearch_list = []

def append_solution(solution, list, env_number, env):
    if solution[0] == None:
        list.append((env_number, solution[2], 0, 0, solution[3], False))
    else:
        list.append([env_number, solution[2], len(solution[1]), env.total_cost(solution[1], scenario=2), solution[3], True])


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
    UCS_list[i][3] = env.total_cost(solution[1], scenario=2)
    solution = ASearch.find_path(env)
    ASearch_list[i][3] = env.total_cost(solution[1], scenario=2)
    
    
print(BFS_list[0])
print(DFS_list[0])
print(DFS_limited_list[0])
print(UCS_list[0])
print(ASearch_list[0])


"""
solution = BFS.find_path(env)
print("BFS:", solution)
solution = DFS.find_path(env)
print("DFS:", solution)
solution = DFS.find_path(env, depth_limit=10)
print("DFS 10 limit:", solution)
solution = UCS.find_path(env)
print("UCS uniforme", solution)
solution = UCS.find_path(env, is_variable=True)
print("UCS variable", solution)
solution = randomAgent.find_path(env)
#print("Random", solution)
solution = ASearch.find_path(env)
print("A*", solution)

env.reset()
env.render()"""