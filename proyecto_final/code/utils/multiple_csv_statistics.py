import pandas as pd
import matplotlib.pyplot as plt
import glob
import ast
import os
import numpy as np

path = "/home/jr/Escritorio/ia-uncuyo-2024/proyecto_final/code/ppo/tests"
files = sorted(glob.glob(f"{path}/*_test.csv"))

def parse_reward(x):
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        return float(ast.literal_eval(x)[0])
    else:
        return float(x)

# Leer y procesar todos los archivos
all_rewards = []
rewards_by_test = {}

for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    print(f"Procesando: {os.path.basename(f)}")
    
    # Asegúrate de usar la columna correcta (puede ser 'reward' o 'episode_reward')
    if 'reward' in df.columns:
        df["reward"] = df["reward"].apply(parse_reward)
        rewards = df["reward"].values
    elif 'episode_reward' in df.columns:
        df["episode_reward"] = df["episode_reward"].apply(parse_reward)
        rewards = df["episode_reward"].values
    else:
        print(f"Advertencia: No se encontró columna de recompensa en {f}")
        continue
    
    label = os.path.basename(f).split("_")[0]
    rewards_by_test[label] = rewards
    all_rewards.extend(rewards)

# ===== 1. HISTOGRAMAS INDIVIDUALES (UN ARCHIVO POR CSV) =====
for label, rewards in sorted(rewards_by_test.items()):
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel("Recompensa", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title(f"Distribución de Recompensas - {label}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Líneas de media y mediana
    mean_val = np.mean(rewards)
    median_val = np.median(rewards)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Media: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, 
               label=f'Mediana: {median_val:.2f}')
    plt.legend()
    
    # Estadísticas en el gráfico
    textstr = f'σ: {np.std(rewards):.2f}\nMin: {np.min(rewards):.2f}\nMax: {np.max(rewards):.2f}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    filename = f"histograma_{label}_ppo.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Histograma guardado como {filename}")
    plt.close()
