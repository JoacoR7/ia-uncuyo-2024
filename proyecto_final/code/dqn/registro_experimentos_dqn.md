## Modelo 1

### Hiperparámetros:
```
policy="CnnPolicy",
env=env,
learning_rate=1e-4,
buffer_size=100_000,
learning_starts=10_000,
batch_size=128,
tau=1.0,
gamma=0.99,
train_freq=4,
gradient_steps=1,
target_update_interval=10_000,
exploration_fraction=0.1,
exploration_final_eps=0.01,
verbose=1
```

### Entorno:
- Sólo se implementó el wrapper Custom Env para recortar la imagen y transformarla en escala de grises.

#### Resultados
<figure style="text-align: center;">
  <img src="./graphics/reward_average_dqn1.png" alt="Promedio de recompensas del DQN" width="60%">
  <figcaption><em>Figura 1. Promedio de recompensas obtenidas por el agente DQN (modelo 1) durante el entrenamiento.</em></figcaption>
</figure>


<figure style="text-align: center;">
  <img src="./graphics/length_average_dqn1.png" alt="Promedio de recompensas del DQN" width="60%">
  <figcaption><em>Figura 2. Promedio de duración (en pasos) del agente DQN (modelo 2) durante el entrenamiento.</em></figcaption>
</figure>
