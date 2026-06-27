import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine Learning & Stats Models
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning (PyTorch Engine)
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Load and Explore Data (EDA)
# =====================================================================
print("--- Step 1: Loading and EDA ---")
df = pd.read_csv('airline-passengers.csv')
df.columns = ['Month', 'Passengers']

# Handle missing values using linear interpolation if any exist
if df['Passengers'].isnull().sum() > 0:
    df['Passengers'] = df['Passengers'].interpolate(method='linear')

# Generate and save EDA Visualizations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(pd.to_datetime(df['Month']), df['Passengers'], color='blue', linewidth=2)
plt.title('Airline Passengers Over Time (Trend & Seasonality)')
plt.xlabel('Year')
plt.ylabel('Passengers')

plt.subplot(1, 2, 2)
sns.histplot(df['Passengers'], kde=True, color='purple')
plt.title('Distribution of Passenger Target Variable')
plt.savefig('eda_charts.png', bbox_inches='tight')
plt.close() 

# =====================================================================
# 2. Preprocess Data & Feature Engineering
# =====================================================================
print("\n--- Step 2: Preprocessing & Feature Engineering ---")

# Scale values for Neural Networks
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Passengers'].values.reshape(-1, 1))

# Convert to supervised learning format (Lag = 12)
def create_supervised(data, look_back=12):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 12
X_scaled, Y_scaled = create_supervised(scaled_data, look_back)

# =====================================================================
# 3. Train-Test Split (80% Train / 20% Test)
# =====================================================================
print("\n--- Step 3: Train-Test Split ---")
total_samples = len(X_scaled)
train_size = int(total_samples * 0.80)
test_size = total_samples - train_size

X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
Y_train, Y_test = Y_scaled[:train_size], Y_scaled[train_size:]

# ARIMA preparation (datetime index and native scale)
df_arima = df.copy()
df_arima['Month'] = pd.to_datetime(df_arima['Month'])
df_arima.set_index('Month', inplace=True)

# Align historical dates to match supervised indices exactly
arima_train = df_arima['Passengers'].iloc[:train_size + look_back]
arima_test = df_arima['Passengers'].iloc[train_size + look_back:]

print(f"Total supervised samples: {total_samples}")
print(f"Train size (80%): {len(X_train)} | Test size (20%): {len(X_test)}")

# =====================================================================
# 4. Model Training & Evaluation (PyTorch + ML + Stats)
# =====================================================================
print("\n--- Step 4: Model Training & Evaluation ---")
results = {}

def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_pred_scaled, model_name):
    y_true = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mape_score(y_true, y_pred)
    
    results[model_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'Pred': y_pred}
    print(f"[{model_name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f} | MAPE: {mape:.2f}%")

# --- 4.1 Linear Regression ---
lr = LinearRegression().fit(X_train, Y_train)
evaluate_model(lr.predict(X_test), 'Linear Regression')

# --- 4.2 XGBoost ---
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42).fit(X_train, Y_train)
evaluate_model(xgb.predict(X_test), 'XGBoost')

# Convert data to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test)

# --- 4.3 Artificial Neural Network (ANN) ---
class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(look_back, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x): return self.net(x)

ann = ANNModel()
optimizer = optim.Adam(ann.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(ann(X_train_t), Y_train_t)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    evaluate_model(ann(X_test_t).numpy(), 'ANN')

# --- 4.4 LSTM in PyTorch ---
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
        self.linear = nn.Linear(8, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

lstm = LSTMModel()
optimizer = optim.Adam(lstm.parameters(), lr=0.01)
X_train_lstm = X_train_t.unsqueeze(2)
X_test_lstm = X_test_t.unsqueeze(2)

for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(lstm(X_train_lstm), Y_train_t)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    evaluate_model(lstm(X_test_lstm).numpy(), 'LSTM')

# --- 4.5 ARIMA ---
arima_fit = ARIMA(arima_train, order=(2, 1, 1)).fit()
arima_pred_raw = arima_fit.forecast(steps=len(arima_test))
arima_scaled = scaler.transform(arima_pred_raw.values.reshape(-1, 1))
evaluate_model(arima_scaled, 'ARIMA')

# =====================================================================
# Save Comparative Visualization
# =====================================================================
y_test_real = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
test_dates = arima_test.index

plt.figure(figsize=(14, 6))

# 1. Plot Actual Test Data as the baseline
plt.plot(test_dates, y_test_real, label='Actual Data (Test Period)', color='black', linewidth=2.5)

# 2. Plot each model's predictions over the test dates
for model in results.keys():
    if model == 'Linear Regression':
        # Solid and thicker line to ensure visibility if overlapping occurs
        plt.plot(test_dates, results[model]['Pred'], label=model, linestyle='-', linewidth=3, alpha=0.7)
    else:
        plt.plot(test_dates, results[model]['Pred'], label=model, linestyle='--', linewidth=1.8, alpha=0.9)

# Visual adjustments for the focused test window
plt.title('Model Comparison - Forecast Evaluation (Test Period Only)', fontsize=14, pad=15)
plt.ylabel('Passengers', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')

# Zooming in tightly on the test date bounds
plt.xlim(test_dates.min(), test_dates.max())

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('model_comparison.png', bbox_inches='tight')
plt.close()

# =====================================================================
# 5. Compare Performance Summary
# =====================================================================
print("\n--- Step 5: Performance Summary Matrix ---")
df_metrics = pd.DataFrame(results).T[['RMSE', 'MAE', 'R2', 'MAPE']].sort_values(by='RMSE')
print(df_metrics)
print("\nExecution complete! Charts saved as 'eda_charts.png' and 'model_comparison.png'.")