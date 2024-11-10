# 2. Proveer las respuestas a los puntos 1,2,5,6,7 de la sección 2.4 (página 52 del ISLRv2).

## 1. For each of parts (a) through (d), indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.

### a) The sample size n is extremely large, and the number of predictors p is small.

Si el tamaño de la muestra es muy grande, los métodos flexibles son mejores ya que se va a ajustar mejor a la complejidad de las relaciones y/o patrones que hay entre los datos.
### b) The number of predictors p is extremely large, and the number of observations n is small.

Cuando tenemos datos de estas características, los métodos flexibles no son tan adecuados, ya que se puede producir el "overfitting" por su capacidad de ajustarse a patrones complejos.
### c) The relationship between the predictors and response is highly non-linear.

En este caso los métodos flexibles son mejores, ya que un método inflexible (como regresión lineal) no se adaptan a relaciones no lineales de forma adecuada.
### d) The variance of the error terms σ^2=Var(ϵ) is extremely high.

Cuando la varianza del error es alta, es decir, que hay mucha dispersión entre los errores, hay mucho ruido. Un método flexible va a tender a realizar un sobreajuste con el ruido, provocando una mala generalización. 

## 2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.
### a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.

### b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

### c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the % change in the US market, the % change in the British market, and the % change in the German market.

| Escenario                          | Tipo de problema | Interés    | n   | p   |
| ---------------------------------- | ---------------- | ---------- | --- | --- |
| a) Salarios de CEO                 | Regresión        | Inferencia | 500 | 3   |
| b) Predicción de éxito de producto | Clasificación    | Predicción | 20  | 13  |
| c) Predicción del tipo de cambio   | Regresión        | Predicción | 52  | 3   |
## 5. What are the advantages and disadvantages of a very flexible (versus a less flexible) approach for regression or classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When might a less flexible approach be preferred?

### Ventajas de flexibilidad:
- Puede capturar relaciones complejas.
- Se puede ajustar a los datos de entrenamiento (menor error).
### Desventajas de flexibilidad:
- Sobreajuste: puede ajustarse muy bien a los datos de entrenamiento, pero luego para los datos desconocidos no hay buenas respuestas.
- Complejidad computacional (tiempo y recursos).
### Ventajas de métodos más inflexibles
- Más simples de implementar y de interpretar.
- Menos riesgo de sobreajuste.
- Más rápido.
### Desventajas de métodos más inflexibles.
- No puede capturar relaciones complejas.
### Cuándo elegir un método más flexible
- La estructura de datos es compleja: relaciones no lineales, o que son complicadas de modelar con un método más simple.
- Cuando la cantidad de observaciones es alta.
### Cuándo elegir un método menos flexible
- Cuando la cantidad de observaciones es baja.
- Cuando se necesita inferencia.
## 6. Describe the differences between a parametric and a non-parametric statistical learning approach. What are the advantages of a parametric approach to regression or classification (as opposed to a nonparametric approach)? What are its disadvantages?

Las diferencias entre ambos enfoques son:
- En un enfoque paramétrico, se asume una forma del modelo, mientras que en el no paramétrico se adapta a la estructura de datos.
- En un enfoque paramétrico, se puede interpretar mejor el modelo ya que se ajusta a una función, mientras que en el no paramétrico es más complejo.
## 7. The table below provides a training data set containing six observations, three predictors, and one qualitative response variable. Suppose we wish to use this data set to make a prediction for Y when X1 = X2 = X3 = 0 using K-nearest neighbors.
![](images/pregunta2.7_islrv2.png)

### a) Compute the Euclidean distance between each observation and the test point, X1 = X2 = X3 = 0.

1. d=3
2. d=2
3. d=$\sqrt{10}$
4. d=$\sqrt{5}$
5. d=$\sqrt{2}$
6. d=$\sqrt{3}$
### b) What is our prediction with K = 1? Why?
Con K=1 (buscamos el vecino más cercano), diríamos que el test es verde, ya que la observación 5 es el punto más cercano según la distancia euclidiana.
### c) What is our prediction with K = 3? Why?
Con K=3 (buscamos los 3 vecinos más cercanos), que en este caso serían las observaciones 5, 6 y 2. Como hay más puntos rojos que verdes, podemos decir que la predicción es rojo.
### d) If the Bayes decision boundary in this problem is highly nonlinear, then would we expect the best value for K to be large or small? Why?

- **K bajo**: Un valor bajo puede hacer que el modelo sea sensible al ruido en los datos, llevando a una sobreajuste (overfitting).
- **K alto**: Un valor grande puede hacer que el modelo sea más robusto, pero también puede llevar a un subajuste (underfitting), ya que incluirá más datos que podrían no ser relevantes para la clasificación.

Teniendo esto en cuenta, si nuestro problema es altamente no lineal, lo mejor sería que K sea bajo, ya que permite capturar complejidades locales de datos. Aunque haya riesgo de sobreajuste, en este caso puede ser mejor ya que las relaciones no lineales pueden ser complejas.