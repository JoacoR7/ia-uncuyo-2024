## Introducción
### Objetivo
El presente informe tiene como objetivo comparar el desempeño de un agente reflexivo simple con el de un agente aleatorio en distintos entornos simulados con diferentes tamaños y tasas de suciedad. La evaluación se centrará en la medida de desempeño (cantidad de suciedad limpiada) y las unidades de tiempo consumidas por puntos de performance.
### Descripción
El problema se plantea en un entorno cuadrado de varios tamaños (desde 2x2 hasta 128x128) en el que se distribuye suciedad de manera aleatoria, variando el porcentaje de suciedad entre 0.1 y 0.8. El agente reflexivo tiene la tarea de limpiar la mayor cantidad de suciedad posible en el menor tiempo posible dentro de las 1000 acciones.
### Metodología
Para cada combinación de tamaño de entorno y porcentaje de suciedad, se realizarán 10 simulaciones. Los resultados de estas simulaciones se analizarán para determinar el rendimiento del agente reflexivo y del agente aleatorio.
## Desarrollo
### Entorno
Se utilizaron entornos de tamaño: 2x2, 4x4, 8x8, 16x16, 32x32, 64x64, y 128x128. Para cada tamaño, se simularon porcentajes de suciedad del 10%, 20%, 40%, y 80%. La suciedad se distribuyó de manera aleatoria en cada simulación. Cada entorno se duplicó para comparar el funcionamiento de ambos agentes en las mismas condiciones.
### Medición del desempeño:  
El desempeño del agente se mide como el porcentaje de suciedad limpiada al final de cada simulación. Adicionalmente, se registra el número de unidades de tiempo consumidas durante la limpieza.
## Análisis y discusión
Durante la ejecución de los agentes, se almacenó el resultado en un archivo .xlsx, donde se almacenaron los siguientes datos:
- Tasa de suciedad
- Cantidad de celdas sucias
- Puntaje máximo de agente reflexivo simple
- Puntaje máximo de agente aleatorio
- Cantidad de acciones realizadas por puntaje
### Comparación de performance
Para obtener información relevante sobre los agentes y su performance, se realizaron 2 tipos de gráficos, el primero muestra qué puntaje alcanzó cada agente de cierto tamaño de entorno en cada tasa de suciedad. El segundo tipo de gráfico utiliza esta información pero se muestra en formato de gráfico de cajas y extensiones para tener una información más comprensible del comportamiento de ambos agentes.

![[comparacionPerformance.png]]

Aquí se puede observar que el agente reflexivo simple obtiene mayor puntaje de performance en cada escenario, en otras palabras, el agente aleatorio tiene un peor desempeño. Y a medida que la tasa de suciedad aumenta, la brecha se agranda más aún.

![[boxNWhiskers.png]]

El gráfico de cajas y extensiones ofrece una visualización clara de la distribución del desempeño de ambos agentes en un entorno de 16x16 con diferentes tasas de suciedad. Se puede observar que la mediana del agente reflexivo simple es mucho más alta en comparación con la del agente aleatorio, esto significa que el agente reflexivo simple tiene un mejor resultado que el aleatorio.

Este gráfico también confirma que, a medida que aumenta la tasa de suciedad, la diferencia en desempeño entre ambos agentes se agranda. 
### Comparación de unidad de tiempo
No sólo sirve tener información de qué puntaje puede obtener cada agente, sino también cuánto tardará en promedio llegar a dichos puntajes. Para ello se realizó un gráfico de barras en el cual se detalla cuántas acciones tomó llegar a cada punto de performance, desde 1 al performance máximo alcanzado dentro del entorno en las distintas tasas de suciedad.

![[comparacionPerformance.png]]

En este caso se grafica la información obtenida para un entorno de tamaño 128x128 con tasa de suciedad de 0.8. Podemos observar que no sólo el agente reflexivo simple obtiene un mayor puntaje, sino que también tarda aproximadamente la mitad que el agente aleatorio en alcanzar ciertos puntos de performance. Se puede observar, por ejemplo, que a un agente  aleatorio le toma, en promedio, alrededor de 800 acciones para obtener 101 puntos de performance, mientras que a un agente reflexivo simple le toma alrededor de 400 acciones.

## Conclusión
El análisis comparativo del desempeño entre un agente reflexivo simple y un agente aleatorio en entornos de distintos tamaños y tasas de suciedad demuestra una clara superioridad del agente reflexivo simple en todos los escenarios evaluados. 

Los gráficos presentados confirman que, en términos de puntaje, el agente reflexivo simple siempre supera al agente aleatorio, y esta ventaja se agranda significativamente en entornos con mayores tasas de suciedad. Esto se traduce en una brecha cada vez más amplia entre ambos agentes.

Además del desempeño en términos de limpieza, el análisis del tiempo necesario para alcanzar ciertos niveles de rendimiento remarca otra ventaja clave del agente reflexivo simple: su eficiencia temporal. Mientras que el agente aleatorio requiere un mayor número de acciones para alcanzar puntajes similares, el agente reflexivo simple logra los mismos resultados en aproximadamente la mitad del tiempo. Esto refuerza la idea de que el agente reflexivo simple no solo es más efectivo, sino también más rápido y eficiente en la realización de su tarea.