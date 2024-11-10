# Proceso de preprocesamiento

1. Eliminación de elementos pertenecientes la clase especie minoritaria (menos de 50 ejemplares).
2. Oversampling
3. Random forest con:
	- Atributos: altura, circ_tronco_cm, diametro_tronco, especie, seccion
	- Número de árboles: 500

# Resultados obtenidos

- Score: 0.71146
# Algoritmo propuesto

Se realizó un modelo con random forest en el cual se predecía la inclinación peligrosa con los atributos de altura, circ_tronco_cm, diametro_tronco, especie y seccion, con una cantidad de 500 árboles. La cantidad de atributos por árbol no se especificó, así que el modelo tomó la cantidad por defecto.

![](vz1f8191.Ensemble-of-decision-trees.png)
