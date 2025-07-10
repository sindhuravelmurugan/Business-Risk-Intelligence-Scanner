import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
import tensorflowjs as tfjs

df = pd.read_csv("/kaggle/input/sme-financial-decision-making/sme_decision.csv")
print("Loaded shape:", df.shape)
print("Columns:", df.columns)

features = ['Revenue', 'Debt_to_Equity', 'Cash_Flow', 'Credit_Score', 'Years_Active', 'Employees']
target = 'Risk_Score'  # Replace with actual risk score column name

df = df.dropna(subset=features + [target])
 
if df[target].max() > 1:
    df[target] = df[target] / 100.0

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaler Mean:", scaler.mean_)
print("Scaler Std :", scaler.scale_)

with open("scaler_stats.js", "w") as f:
    f.write("const scaler = {\n")
    f.write(f"  mean: {list(scaler.mean_)},\n")
    f.write(f"  std: {list(scaler.scale_)}\n")
    f.write("};\n")
    f.write("export default scaler;")

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')  # Linear for regression output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

model.save("business_risk_model.h5")

os.makedirs("tfjs_model", exist_ok=True)
tfjs.converters.save_keras_model(model, "tfjs_model")

print("Model training + TF.js export complete.")
