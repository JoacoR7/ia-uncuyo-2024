import BFS
import DFS
import UCS
import environment

env = environment.Environment(8, 100, 0.2)

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

env.reset()
env.render()