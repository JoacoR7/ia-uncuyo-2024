import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("dqn5.csv")

# Crear columna de episodio
df["episode"] = df.index + 1

# Media móvil (puede ajustarse)
window = 100

# Suavizados
df["reward_smooth"] = df["episode_reward"].rolling(window=window).mean()
df["length_smooth"] = df["episode_length"].rolling(window=window).mean()

# --- Gráfica 1: Recompensa suavizada ---
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["reward_smooth"], linewidth=2)
plt.xlabel("Episodio")
plt.ylabel("Recompensa (media móvil)")
plt.title(f"Recompensa suavizada por episodio (MA={window})")
plt.savefig("reward_average_dqn5.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Gráfica 2: Duración suavizada ---
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["length_smooth"], linewidth=2)
plt.xlabel("Episodio")
plt.ylabel("Duración del episodio (media móvil)")
plt.title(f"Duración suavizada por episodio (MA={window})")
plt.savefig("length_average_dqn5.png", dpi=300, bbox_inches="tight")
plt.close()
