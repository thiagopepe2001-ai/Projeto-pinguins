# %%
import seaborn as sns

# %%Carrega o dataset instantaneamente
df = sns.load_dataset("penguins")
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# %%1. Carrega o dataset de pinguins
df = sns.load_dataset("penguins")

# 2. Exibe as primeiras 5 linhas
print("--- Primeiras Linhas do Dataset ---")
print(df.head())

# 3. Exibe informações sobre colunas e valores nulos
print("\n--- Informações do Dataset ---")
print(df.info())
import pandas as pd
import seaborn as sns

# 1. Carrega o dataset original do Seaborn
df_original = sns.load_dataset("penguins")
print(f"Total de linhas original: {len(df_original)}")

# 2. Exclui todas as linhas com valores nulos (NaN)
df_limpo = df_original.dropna()
print(f"Total de linhas após a limpeza: {len(df_limpo)}")
print(f"Linhas removidas: {len(df_original) - len(df_limpo)}\n")

from sklearn.model_selection import train_test_split


# 3. Exibe o resumo para confirmar que todas as colunas têm 333 linhas preenchidas
print("--- Informações do Dataset Limpo ---")
print(df_limpo.info())

# Utilizando o dataset limpo
X = df_limpo.drop(columns=["species"])
y = df_limpo["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dados de Treino: {X_train.shape[0]} linhas")
print(f"Dados de Teste: {X_test.shape[0]} linhas\n")


# Separa as características (X)
X = df_limpo.drop(columns=["species"])

# Separa a variável que queremos prever (y)
y = df_limpo["species"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Dados de Treino: {X_train.shape[0]} linhas")
print(f"Dados de Teste: {X_test.shape[0]} linhas")


# Importando as ferramentas de pré-processamento

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


# Colunas categóricas
colunas_categoricas = ["island", "sex"]

# Colunas numéricas
colunas_numericas = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]


preprocessador = ColumnTransformer(
    transformers=[
        ("categoricas", OneHotEncoder(), colunas_categoricas),
        ("numericas", StandardScaler(), colunas_numericas)
    ]
)


# Aprende as transformações com os dados de treinamento
X_train_transformado = preprocessador.fit_transform(X_train)

# Aplica as transformações nos dados de teste
X_test_transformado = preprocessador.transform(X_test)


print("--- Dados Originais ---")
print("Treinamento:", X_train.shape)
print("Teste:", X_test.shape)

print("\n--- Dados Transformados ---")
print("Treinamento:", X_train_transformado.shape)
print("Teste:", X_test_transformado.shape)


from sklearn.ensemble import RandomForestClassifier


modelo = Pipeline(steps=[
    ("preprocessamento", preprocessador),

    ("classificador", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])



modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)

print("Previsões do modelo:")
print(y_pred[:10])

print("\nEspécies reais:")
print(y_test.iloc[:10].values)


from sklearn.metrics import accuracy_score

acuracia = accuracy_score(y_test, y_pred)

print(f"Acurácia do modelo: {acuracia:.2%}")

import joblib

modelo.fit(X_train, y_train)


joblib.dump(modelo, "modelo_pinguins.pkl")