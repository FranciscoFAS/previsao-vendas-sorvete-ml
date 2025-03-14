import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import mlflow
import mlflow.sklearn
import pickle
from fastapi import FastAPI
import uvicorn

df = pd.read_csv('sorvete_vendas.csv', parse_dates=['data'])


df['dia'] = df['data'].dt.day
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year

plt.figure(figsize=(12, 6))

# Scatter Plot (Temperatura x Vendas)
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['temperatura'], y=df['vendas_sorvete'])
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete')
plt.title('Temperatura x Vendas')

# Linha Temporal (Data x Vendas)
plt.subplot(1, 2, 2)
sns.lineplot(x=df['data'], y=df['vendas_sorvete'])
plt.xlabel('Data')
plt.ylabel('Vendas de Sorvete')
plt.title('Tendência de Vendas ao Longo do Tempo')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

X = df[['temperatura', 'dia', 'mes', 'ano']]
y = df['vendas_sorvete']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
erro = mean_absolute_error(y_test, y_pred)
print(f'Erro médio absoluto: {erro:.2f}')

mlflow.set_experiment('previsao_vendas_sorvete')
with mlflow.start_run():
    mlflow.log_param('modelo', 'Regressao Linear')
    mlflow.log_metric('MAE', erro)
    mlflow.sklearn.log_model(modelo, 'modelo_sorvete')

# Salvar modelo para deploy
with open('models/modelo_sorvete.pkl', 'wb') as f:
    pickle.dump(modelo, f)

app = FastAPI()

@app.get("/prever/")
def prever(temperatura: float, dia: int, mes: int, ano: int):
    modelo = pickle.load(open('models/modelo_sorvete.pkl', 'rb'))
    previsao = modelo.predict([[temperatura, dia, mes, ano]])
    return {"previsao_vendas": int(previsao[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
