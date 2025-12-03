import pandas as pd
import matplotlib.pyplot as plt
import glob
import ast
import os

path = "/home/jr/Escritorio/ia-uncuyo-2024/proyecto_final/code/ppo/tests"
files = sorted(glob.glob(f"{path}/*_test.csv"))

def parse_reward(x):
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        return float(ast.literal_eval(x)[0])
    else:
        return float(x)

plt.figure(figsize=(10,6))
window = 100

for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    print(f"Procesando: {os.path.basename(f)}")

    df["episode_length"] = df["episode_length"].apply(parse_reward)
    df["moving_avg"] = df["episode_length"].rolling(window).mean()

    label = os.path.basename(f)
    label = label.split("_")[0]
    plt.plot(df["episode"], df["moving_avg"], label=label)
    

plt.xlabel("Episodio")
plt.ylabel("Pasos")
plt.title(f"Media móvil (SMA{window}) de pasos de los test PPO en 1000 episodios")
plt.legend()
plt.grid(True)
plt.savefig("pasos_test_ppo.png", dpi=300, bbox_inches="tight")
print("Gráfico guardado como plot.png")
