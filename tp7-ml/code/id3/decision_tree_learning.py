import pandas
import numpy

# Atributo más frecuente
def plurality_value(data, column):
    # mode() devuelve una lista con los valores que aparecen con mayor frecuencia en la columna
    # elegimos el primero ya que pueden haber varios con la misma frecuencia
    return data[column].mode()[0]

# Mide la incertidumbre de una variable aleatoria, mientras más baja la entropía,
# menor información se requiere para predecir un resultado
def entropy(column):
    entropy_value = 0
    n = len(column)
    values = column.value_counts()
    for i in values:
        # P(vk) es la probabilidad de que la variable aleatoria V tome el valor vk
        p = i/n
        entropy_value -= p * numpy.log2(p)
    return entropy_value

def importance(examples, attribute, target_column):
    # Calcular la entropía total del conjunto de ejemplos para la columna objetivo
    total_entropy = entropy(examples[target_column])
    
    # Calcular la entropía ponderada de cada valor del atributo
    weighted_entropy = 0
    for value in examples[attribute].unique():
        subset = examples[examples[attribute] == value]
        prob = len(subset) / len(examples)
        weighted_entropy += prob * entropy(subset[target_column])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def decision_tree_learning(examples, attributes, parent_examples, target_column):
    """
    Construye un árbol de decisión a partir de un conjunto de ejemplos y atributos.
    
    Parámetros:
    examples (pandas.DataFrame): Conjunto de ejemplos de entrenamiento.
    attributes (list): Lista de atributos disponibles.
    parent_examples (pandas.DataFrame): Ejemplos de los padres (para calcular la ganancia de información).
    target_column (str): Nombre de la columna objetivo.
    
    Devuelve:
    Un árbol de decisión representado como un diccionario.
    """
    if examples.empty:
        return plurality_value(parent_examples, target_column)
    elif len(examples[target_column].unique()) == 1:
        return examples[target_column].iloc[0]
    elif not attributes:
        return plurality_value(examples, target_column)
    else:
        best_attribute = max(attributes, key=lambda a: importance(examples, a, target_column))
        tree = {best_attribute: {}}
        for value in examples[best_attribute].unique():
            subset = examples[examples[best_attribute] == value]
            subtree = decision_tree_learning(subset, [attr for attr in attributes if attr != best_attribute], examples, target_column)
            tree[best_attribute][value] = subtree
        return tree