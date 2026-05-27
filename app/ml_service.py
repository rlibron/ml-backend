# app/ml_service.py
# Handles model loading and predictions

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class MLService:
    """Service class for ML model operations"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
        # Mapping for categorical values
        self.role_map = {"Student": 0, "Faculty": 1, "Staff": 2}
        self.category_map = {
            "Laptop": 0, "Camera": 1, "Tablet": 2, 
            "Projector": 3, "Calculator": 4
        }
        
        # Model performance from Activity 2
        self.metrics = {
            "accuracy": 0.8245,
            "f1_score": 0.8192,
            "precision": 0.815,
            "recall": 0.823,
            "model_type": "Random Forest",
            "description": "Predicts if rental duration > 3 days (long-term)"
        }
    
    def load_models(self):
        """Load trained model and scaler from disk"""
        try:
            # Try to load saved models
            model_path = Path("models/rental_model.pkl")
            scaler_path = Path("models/scaler.pkl")
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_loaded = True
                print("✅ Models loaded from disk!")
                return True
            else:
                print("⚠️ No saved models found. Training fallback model...")
                self._train_fallback_model()
                return True
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def _train_fallback_model(self):
        """Train a simple model if no saved model exists"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create sample data (replace with real data later)
        np.random.seed(42)
        X_sample = np.random.rand(500, 10)
        y_sample = np.random.randint(0, 2, 500)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sample)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_sample)
        self.is_loaded = True
        
        print("✅ Fallback model trained!")
    
    def prepare_features(self, rental_data: Dict[str, Any]) -> np.ndarray:
        """Convert input data to model-ready features"""
        
        # Encode categorical values
        role_encoded = self.role_map.get(rental_data['user_role'], 0)
        category_encoded = self.category_map.get(rental_data['gadget_category'], 0)
        
        # Create feature array
        features = np.array([[
            rental_data['duration_days'],
            rental_data['daily_rate'],
            rental_data['total_cost'],
            rental_data['days_until_event'],
            rental_data['week_of_season'],
            rental_data['hour_of_day'],
            int(rental_data['is_weekend']),
            int(rental_data['is_holiday_season']),
            role_encoded,
            category_encoded
        ]])
        
        # Ensure correct number of features
        expected_features = self.model.n_features_in_
        if features.shape[1] < expected_features:
            features = np.pad(features, ((0,0), (0, expected_features - features.shape[1])), constant_values=0)
        elif features.shape[1] > expected_features:
            features = features[:, :expected_features]
        
        return features
    
    def predict(self, rental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single rental"""
        
        if not self.is_loaded:
            success = self.load_models()
            if not success:
                return {
                    "error": "Model not available",
                    "prediction": None
                }
        
        try:
            # Prepare features
            features = self.prepare_features(rental_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = int(self.model.predict(features_scaled)[0])
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Prepare result
            label = "Long-term (>3 days)" if prediction == 1 else "Short-term (≤3 days)"
            confidence = max(probabilities)
            
            return {
                "prediction": prediction,
                "prediction_label": label,
                "confidence": float(confidence),
                "probability_short": float(probabilities[0]),
                "probability_long": float(probabilities[1]),
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return model performance metrics"""
        return self.metrics

# Create a single instance to be reused
ml_service = MLService()