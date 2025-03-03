
# Aprendizaje por refuerzo para ATARI - Space Invaders <!-- omit in toc -->
---
Código de proyecto: SPACEAI

---
## Índice <!-- omit in toc -->

- [Introducción](#introducción)
- [Marco teórico](#marco-teórico)
  - [OpenAI Gymnasium API](#openai-gymnasium-api)
    - [Características](#características)
    - [Funcionamiento](#funcionamiento)

## Introducción
(A completar)

## Marco teórico
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

