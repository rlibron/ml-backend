# train_model.py
# Train your model from Activity 2 and save it

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING MODEL FOR GADGET RENTAL PREDICTION")
print("="*60)

# Load your dataset
df = pd.read_csv('gadget_rental.csv')
print(f"Loaded {len(df)} records")

# Create target variable (from Activity 2)
median_duration = df['duration_days'].median()
df['target'] = (df['duration_days'] > median_duration).astype(int)
print(f"Target created: {df['target'].value_counts().to_dict()}")

# Preprocessing (same as Activity 2)
print("\nPreprocessing data...")

# Handle missing values
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Select features (same as Activity 2)
exclude_cols = ['rental_ID', 'user_ID', 'gadget_ID', 'rental_date', 'return_date', 
                'event_start_date', 'event_end_date', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print(f"\nTraining Random Forest on {X_scaled.shape[1]} features...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# Evaluate
from sklearn.metrics import accuracy_score, f1_score
y_pred = model.predict(X_scaled)
print(f"\nTraining Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Training F1-Score: {f1_score(y, y_pred, average='weighted'):.4f}")

# Save model and scaler
import os
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/rental_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n✅ Model saved to: models/rental_model.pkl")
print("✅ Scaler saved to: models/scaler.pkl")

# Test prediction
print("\n" + "="*60)
print("TESTING SAVED MODEL")
print("="*60)

# Load and test
loaded_model = joblib.load('models/rental_model.pkl')
loaded_scaler = joblib.load('models/scaler.pkl')

sample = X_scaled[0:1]
pred = loaded_model.predict(sample)
print(f"Sample prediction: {'Long-term' if pred[0] == 1 else 'Short-term'}")

print("\n🎉 Model training complete! Ready for FastAPI.")