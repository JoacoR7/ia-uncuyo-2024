
# Aprendizaje por refuerzo para ATARI - Space Invaders <!-- omit in toc -->
---
Código de proyecto: SPACEAI

---
## Índice <!-- omit in toc -->

- [Introducción](#introducción)
- [Marco teórico](#marco-teórico)
  - [Reinforcement Learning](#reinforcement-learning)
    - [Q-learning](#q-learning)
    - [Deep Q-Network](#deep-q-network)
  - [OpenAI Gymnasium API](#openai-gymnasium-api)
    - [Características](#características)
    - [Funcionamiento](#funcionamiento)

## Introducción
(A completar)

## Marco teórico

### Reinforcement Learning

El **Reinforcement Learning (RL)** o **aprendizaje por refuerzo** es un paradigma del aprendizaje automático en el que un agente aprende a tomar decisiones en un entorno para maximizar una recompensa acumulada. En RL, el agente interactúa con el entorno siguiendo un proceso de prueba y error, utilizando una política que define qué acción tomar en cada estado.  

Este enfoque está inspirado en cómo los animales aprenden mediante **ensayo y error**, utilizando **recompensas positivas y negativas**. Por ejemplo, al entrenar a un perro para realizar trucos, se le da un premio como refuerzo positivo cuando ejecuta correctamente una acción. De manera similar, un agente de RL aprende a comportarse de forma óptima en un entorno al recibir recompensas o penalizaciones según sus acciones.  

El aprendizaje en RL se basa en los siguientes elementos clave:  

- **Agente:** Es el sistema que toma acciones.
- **Entorno:** Es el espacio en el que opera el agente.
- **Estado (S):** Representa la situación actual del agente en el entorno.
- **Acciones (A):** Conjunto de decisiones que el agente puede tomar.
- **Recompensa (R):** Es un valor que recibe el agente al realizar una acción, nos indica que tan buena fue la decisión del agente.
- **Política (π):** Estrategia que define que acción tomar en cada estado.

El objetivo del agente es aprender una política óptima $π^*$ que maximice la suma de recompensas a lo largo del tiempo. Para lograrlo, se utilizan diferentes algoritmos de aprendizaje, como **Q-learning, Deep Q-Network (DQN) y Double Deep Q-Network (DDQN)**.

### Q-learning
**Q-learning** es un algoritmo de aprendizaje por refuerzo basado en valores, cuyo objetivo es aprender una función de acción-valor **Q(s, a)** , que representa la recompensa esperada si el agente toma la acción **a** en el estado **s** y sigue la política óptima a partir de ahí.  

El algoritmo actualiza iterativamente la función **Q(s, a)** mediante la ecuación de Bellman: 

$
Q(s, a) \leftarrow Q(s, a) + α \left[ R + \gamma \max_{a'} Q(s', a')  - Q(s, a) \right]
$

Donde:  

- α es la tasa de aprendizaje (*learning rate*).  
- γ es el factor de descuento, que pondera la importancia de futuras recompensas.  
- R es la recompensa recibida al ejecutar la acción a.  
- s' es el nuevo estado tras la acción a .  
- $( max_{a'} Q(s', a') )$ representa el valor máximo esperado desde el nuevo estado.  


El algoritmo de Q-learning, bajo ciertas condiciones (como una tasa de aprendizaje adecuada y la exploración suficiente), converge a una **política óptima**. La política óptima es la que maximiza la recompensa esperada a largo plazo para el agente. Es importante notar que Q-learning es un algoritmo **off-policy**, lo que significa que el agente puede aprender la política óptima sin tener que seguir exactamente la política que está aprendiendo. 

En resumen, Q-learning es un algoritmo de aprendizaje por refuerzo eficiente que permite a un agente aprender una política óptima de acción para maximizar recompensas a largo plazo. Aunque es un algoritmo potente e **independiente del modelo**, lo que significa que no necesita conocer el entorno de antemano, su rendimiento puede ser limitado en entornos con espacios de estados grandes o continuos. La principal ventaja es su capacidad de aprender sin necesidad de un modelo explícito del entorno, pero su **lentitud de convergencia** en problemas complejos y la necesidad de adaptaciones, como las redes neuronales en **Deep Q-Learning**, son algunas de sus principales limitaciones. Además, Q-learning depende de un adecuado balance entre **exploración y explotación**, lo que puede ser un desafío en ciertos contextos.

### Deep Q-Network 

El **Deep Q-Learning** es una extensión del algoritmo clásico Q-learning que utiliza redes neuronales profundas para aproximar la función de valores **Q(s, a)** en entornos de alta dimensión y con espacios de estados complejos. A diferencia de Q-learning, que emplea una tabla explícita para almacenar los valores de Q, **Deep Q-Learning** utiliza una red neuronal para predecir estos valores, lo que permite manejar escenarios donde los estados no son discretos o son demasiado numerosos para almacenar en una tabla.

El **Deep Q-Network (DQN)** es una implementación específica de **Deep Q-Learning** que introduce mejoras clave para garantizar la estabilidad y eficiencia del aprendizaje. DQN incorpora técnicas como el **Replay Buffer**, que almacena experiencias pasadas para romper la correlación temporal entre las muestras y permitir una actualización más robusta de la red, y la **Target Network**, una red neuronal separada que se actualiza con menor frecuencia para evitar oscilaciones durante el entrenamiento.

DQN ha sido una de las innovaciones más importantes en **Reinforcement Learning**, permitiendo aplicar **Q-learning** en entornos con espacios de estados continuos y de alta dimensión.

#### **Arquitectura de la Red Neuronal en DQN**  

DQN emplea una **red neuronal artificial** con tres componentes principales:  

- **Capa de entrada**: Recibe la representación del estado actual del entorno s , que puede ser una imagen (como en juegos de Atari) o una serie de valores numéricos.  
- **Capas ocultas**: Son capas intermedias con múltiples neuronas que extraen características relevantes del estado. Utilizan funciones de activación no lineales, como **ReLU (Rectified Linear Unit)**, para capturar patrones complejos.  
- **Capa de salida**: Genera un conjunto de valores Q(s, a), donde cada neurona en esta capa representa el valor estimado de tomar una acción específica a en el estado s.  

#### **Aprendizaje y Optimización**

Para mejorar la precisión de los valores Q(s, a), DQN ajusta los **pesos** de las conexiones entre neuronas mediante el algoritmo de **backpropagation** y optimización por **descenso de gradiente estocástico (SGD)** o variantes como **Adam**.  

El entrenamiento de DQN incluye las siguientes técnicas clave:  

- **Replay Buffer**: Almacena experiencias pasadas (s, a, r, s') en un buffer y las reutiliza para entrenar la red, reduciendo la correlación entre muestras consecutivas.  
- **Target Network**: Utiliza una segunda red neuronal (red objetivo) para calcular los valores Q en la actualización, evitando oscilaciones inestables durante el entrenamiento.  
- **Exploración con  ϵ-greedy**: Equilibra la exploración y explotación ajustando la probabilidad de elegir acciones aleatorias a medida que el agente aprende.


---
### OpenAI Gymnasium API

Gymnasium es una librería diseñada para desarrollar y evaluar algoritmos de aprendizaje por refuerzo (RL). Proporciona una interfaz estandarizada que facilita la creación de agentes de RL y su entrenamiento.

#### Características
- **Interfaz unificada:** Proporciona una estructura estándar para interactuar con cualquier entorno con funciones como render(), step() y reset().
- **Variedad de entornos:** Da la posibilidad de interactuar con simulaciones físicas y también videojuegos clásicos de Atari como Space Invaders, el que vamos a tratar en este proyecto.
- **Compatibilidad con librerías:** No impone el uso de ninguna librería, por lo tanto se pueden usar librerías como Stable Baselines, TensorFlow y PyTorch, que es la que usaremos en este proyecto.

#### Funcionamiento
Para la interacción con un entorno de Gymnasium, el proceso es el siguiente:

1. **Inicializar el entorno:** Se crea el entorno con gym.make('ALE/SpaceInvaders-v5'), lo que permite interactuar con el juego.
2. **Reiniciar el entorno:** Se usa env.reset(), lo que devuelve el estado inicial del juego.
3. **Tomar acciones:** En cada paso, se elige una acción (como moverse a la izquierda, derecha o disparar) y se ejecuta con env.step(action).
4. **Observar el resultado:** El entorno devuelve cuatro elementos clave:
- Observación: Imagen del juego después de la acción.
- Recompensa: Puntos obtenidos en ese paso.
- Done: Indica si el juego terminó.
- Info: Datos adicionales como puntaje acumulado.

