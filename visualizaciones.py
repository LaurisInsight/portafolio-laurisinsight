1️⃣ Setup: importar librerías y cargar datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Opcional: estilo prolijo
sns.set(style="whitegrid")

# Cargar dataset
df = pd.read_csv("telco_churn.csv")

# Mapear Churn a 0/1 para algunas cosas (si no lo hiciste ya)
df["Churn_flag"] = df["Churn"].map({"No": 0, "Yes": 1})

2️⃣ Distribución de churn

Barrita simple para mostrar cuántos se van y cuántos se quedan.

plt.figure(figsize=(5,4))
ax = sns.countplot(data=df, x="Churn")
plt.title("Distribución de clientes (Churn vs No Churn)")
plt.xlabel("Churn")
plt.ylabel("Cantidad de clientes")

# Mostrar porcentaje arriba de cada barra
total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = 100 * height / total
    ax.annotate(f"{pct:.1f}%",
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

3️⃣ Churn por tipo de contrato

Acá se ve clarísimo que mes a mes suele tener más churn.

# Calcular tasa de churn por tipo de contrato
contract_churn = (df.groupby("Contract")["Churn_flag"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Churn_flag": "Churn_rate"}))

plt.figure(figsize=(6,4))
ax = sns.barplot(data=contract_churn, x="Contract", y="Churn_rate")
plt.title("Tasa de churn por tipo de contrato")
plt.xlabel("Tipo de contrato")
plt.ylabel("Tasa de churn")

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

4️⃣ Cargos mensuales por churn (boxplot)

Sirve para mostrar que los que churnean suelen tener ciertas características de precio.

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Cargos mensuales según churn")
plt.xlabel("Churn")
plt.ylabel("Cargos mensuales")
plt.tight_layout()
plt.show()

5️⃣ Tenure (antigüedad) por churn

Histograma side by side o en la misma figura.

plt.figure(figsize=(7,4))
sns.histplot(data=df, x="tenure", hue="Churn", kde=False, bins=30, multiple="stack")
plt.title("Distribución de antigüedad (tenure) según churn")
plt.xlabel("Meses de antigüedad (tenure)")
plt.ylabel("Cantidad de clientes")
plt.tight_layout()
plt.show()

6️⃣ Mapa de calor de correlaciones numéricas

Bonito para mostrar “foto” cuantitativa del dataset.

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(8,6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=False, cmap="viridis")
plt.title("Mapa de calor de correlaciones (variables numéricas)")
plt.tight_layout()
plt.show()
