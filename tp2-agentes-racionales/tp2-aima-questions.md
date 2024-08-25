### 2.10 (a) ¿Puede un agente reflexivo simple ser perfectamente racional en este entorno? 

En un entorno donde el agente es penalizado por cada movimiento, un agente reflexivo simple (que opera basándose solo en la percepción actual) no podría ser perfectamente racional. Un agente reflexivo simple suele tener reglas fijas que responden a la percepción actual sin considerar el costo de moverse. En este entorno, el costo de moverse debe ser considerado en la estrategia de limpieza. Por ejemplo:
- Si el agente siempre se mueve de manera aleatoria, podría incurrir en muchos movimientos innecesarios, acumulando penalizaciones que reducen su puntuación total.
- El agente reflexivo simple puede no optimizar sus movimientos para minimizar el costo total, lo cual es muy importante en un entorno donde cada movimiento tiene un costo.

### 2.10 (b) ¿Qué pasa con un agente reflexivo con estado? 

Un agente reflexivo con estado tiene una mejor capacidad para ser racional en este entorno, ya que puede utilizar información sobre el estado pasado para tomar decisiones basadas en la información que tiene.

- **Diseño del Agente Reflexivo con Estado**:
    - **Percepción**: El agente observa su ubicación actual y el estado de la suciedad, y mantiene un registro de sus movimientos anteriores.
    - **Estado Interno**: El agente mantiene un mapa o un registro de las áreas visitadas y la suciedad detectada.
    - **Acción**:
        - Si detecta suciedad en la ubicación actual, limpia.
        - Si no hay suciedad y no hay información previa sobre otras ubicaciones con suciedad, el agente busca la dirección con la menor penalización esperada en función del mapa de estado.
        - Utiliza estrategias como moverse solo a ubicaciones que han demostrado ser productivas en el pasado, o utilizar un algoritmo de búsqueda eficiente para minimizar el número de movimientos.

### 2.10 (c) ¿Cómo cambian tus respuestas a y b si las percepciones del agente le dan el estado de limpio/sucio de cada cuadrado en el entorno?

Si el agente contiene la información del estado de cada cuadrado en el entorno, la capacidad de ambos tipos de agentes para ser racionales mejora considerablemente.

**Para el Agente Reflexivo Simple**:
Con información completa sobre la suciedad en cada cuadrado, el agente reflexivo simple puede ajustar sus reglas para moverse solo a ubicaciones que contienen suciedad. Sin embargo, sigue sin considerar el costo de los movimientos, por lo que su racionalidad sigue estando limitada en términos de optimizar el costo total.
**Para el Agente Reflexivo con Estado**:
Con información completa sobre el estado de cada cuadrado, el agente reflexivo con estado puede planificar sus movimientos de manera más eficiente, minimizando el costo total de los movimientos. Puede utilizar estrategias de planificación más avanzadas, como el algoritmo de Dijkstra o A*, para optimizar el recorrido y minimizar las penalizaciones, haciendo que el agente sea más racional.

### 2.11 (a) ¿Puede un agente reflexivo simple ser perfectamente racional en este entorno? 

En el entorno modificado, donde la geografía y la configuración inicial de la suciedad son desconocidas, un agente reflexivo simple (que opera basándose en la percepción actual y un conjunto de reglas predefinidas) no será racional. 

La falta de conciencia del estado y la adaptabilidad del agente limitan su capacidad para desempeñarse de manera óptima en el entorno: como no sabe dónde están los obstáculos, tratará de pasar sobre ellos y eso ya descontará puntos de performance.

### 2.11 (b) ¿Puede un agente reflexivo simple con una función de agente aleatorio superar a un agente reflexivo simple? 

No podría ya que hay situaciones en las cuales el agente se colocaría en la celda sucia y podría simplemente seguir en vez de limpiar, y eso costaría puntos de performance.

### 2.11 (d) ¿Puede un agente reflexivo con estado superar a un agente reflexivo simple? 

Un agente reflexivo con estado puede superar a un agente reflexivo simple porque mantiene alguna memoria de acciones y observaciones pasadas, permitiéndole tomar decisiones más informadas.