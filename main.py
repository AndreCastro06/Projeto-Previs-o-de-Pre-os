import yfinance as financesource
import pandas as pd
import numpy as np
import matplotlib.pyplot as matplt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import ta  

# Consulta de historico das Ações AAPL (Apple) e MSFT (Microsoft)
tickers = ["AAPL", "MSFT"]  
data = financesource.download(tickers, start="2024-01-01", end="2025-01-01")

# Relacionar apenas preços de fechamento
data = data['Close']

# Calcular Indicadores Técnicos (Média Móvel, RSI, MACD)

# Média Móvel Simples (14 dias)
data['AAPL_MA'] = ta.trend.sma_indicator(data['AAPL'], window=30)

  # Média Móvel Simples (14 dias)
data['MSFT_MA'] = ta.trend.sma_indicator(data['MSFT'], window=30)

data['AAPL_RSI'] = ta.momentum.rsi(data['AAPL'], window=14)  # Relative Strength Index (RSI)
data['MSFT_RSI'] = ta.momentum.rsi(data['MSFT'], window=14)

data['AAPL_MACD'] = ta.trend.macd(data['AAPL'])  # MACD
data['MSFT_MACD'] = ta.trend.macd(data['MSFT'])

# Remover NaN
data = data.dropna()

# Historico de Valores
matplt.figure(figsize=(12,6))
matplt.plot(data.index, data['AAPL'], label='AAPL', color='blue')
matplt.plot(data.index, data['MSFT'], label='MSFT', color='orange')
matplt.title('Historico de Valores')
matplt.legend()
matplt.show()

# Preparação dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length=30):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(data_scaled, seq_length=30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

input_size = X_train.shape[2]
hidden_layer_size = 50
output_size = y_train.shape[1]

model = LSTMModel(input_size, hidden_layer_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f"Test Loss: {test_loss.item()}")

# Converter previsões e valores reais para escala original
y_test_original = scaler.inverse_transform(y_test)
predictions_original = scaler.inverse_transform(predictions.numpy())

# Plotar previsões vs valores reais
matplt.figure(figsize=(12,6))
matplt.plot(y_test_original[:, 0], label='Real AAPL', color='blue')
matplt.plot(predictions_original[:, 0], label='Previsto AAPL', color='cyan', linestyle='dashed')
matplt.plot(y_test_original[:, 1], label='Real MSFT', color='orange')
matplt.plot(predictions_original[:, 1], label='Previsto MSFT', color='red', linestyle='dashed')
matplt.title('Comparação entre Preço Real x Previsão')
matplt.legend()
matplt.show()

print("Projetinho On! 🚀")
