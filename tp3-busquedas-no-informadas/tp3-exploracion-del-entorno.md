## 4)
### a)
La función is_slippery tiene como predeterminado el valor True, esto controla justamente si el lago "está resbaloso". Lo que tiene este efecto es que el personaje tiene la posibilidad de moverse de forma perpendicular al movimiento deseado, es decir, si quiere moverse para abajo, pueda que se mueva para la izquierda o derecha.
### b)
- Primer entorno
	`env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", render_mode='human')`
	Tiene un tamaño de 4x4, y contiene 4 agujeros. La posición inicial del agente es (0,0), y la del objetivo (3,3).
- Segundo entorno
	`desc=["SFFF", "FHFH", "FFFH", "HFFG"]`
	`env = gym.make('FrozenLake-v1', desc=desc, render_mode='human')`
	Tiene la misma configuración que el primer entorno.
- Tercer entorno
	`env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), render_mode='human')`
	Tiene un tamaño de 8x8, la cantidad de agujeros es aleatoria, pero la posición inicial del personaje es siempre (0,0) y la del objetivo (7x7).
