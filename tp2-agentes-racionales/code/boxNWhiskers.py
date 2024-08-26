import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar el archivo Excel
file_path = r'C:\Users\joaqu\Desktop\Análisis agentes racionales.xlsx'
sheet_name = "16x16"
df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

# Extracción de datos
performance_simple = df.iloc[2, 1:41]
performance_random = df.iloc[3, 1:41]

# Verificar longitud
print(f'Longitud de performance_simple: {len(performance_simple)}')
print(f'Longitud de performance_random: {len(performance_random)}')

# Crear el DataFrame con los datos para seaborn
# Crear un DataFrame para cada tipo de agente
df_simple = pd.DataFrame({
    'Tasa de Suciedad': ["0.1"]*10 + ["0.2"]*10 + ["0.4"]*10 + ["0.8"]*10,
    'Performance': performance_simple
})

df_random = pd.DataFrame({
    'Tasa de Suciedad': ["0.1"]*10 + ["0.2"]*10 + ["0.4"]*10 + ["0.8"]*10,
    'Performance': performance_random
})

# Añadir una columna para diferenciar los agentes
df_simple['Agente'] = 'Agente reflexivo simple'
df_random['Agente'] = 'Agente aleatorio'

# Concatenar los DataFrames
data = pd.concat([df_simple, df_random], ignore_index=True)

# Crear el gráfico de cajas y bigotes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Tasa de Suciedad', y='Performance', hue='Agente', data=data)
plt.title('Gráficos de caja y extensiones de performance de agentes reflexivo simple y random: Entorno ' + sheet_name)
plt.xlabel('Tasa de Suciedad')
plt.ylabel('Puntaje de performance')
plt.show()
