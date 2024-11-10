import pandas
import decision_tree_learning

def print_decision_tree(tree, level=0, prefix=""):
    """
    Imprime un árbol de decisión en un formato legible y estructurado.

    Parámetros:
    tree (dict): El árbol de decisión representado como un diccionario.
    level (int): El nivel actual del árbol (utilizado para la indentación).
    prefix (str): Prefijo utilizado para la visualización de la jerarquía.
    """
    if isinstance(tree, dict):
        for attribute, subtree in tree.items():
            print(f"{prefix}├── {attribute}")
            # Llamada recursiva para imprimir el subárbol
            new_prefix = prefix + "│   "
            print_decision_tree(subtree, level + 1, new_prefix)
    else:
        print(f"{prefix}└── -> {tree}")


if __name__ == "__main__":
    csv = pandas.read_csv("https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/tennis.csv")
    
    attributes = list(csv.columns)
    attributes.remove("play")
    target = "play"

    tree = decision_tree_learning.decision_tree_learning(csv, attributes, csv, target)
    print_decision_tree(tree)