# app/config.py
# Configuration settings for the API

class Settings:
    APP_NAME = "Gadget Rental ML API"
    APP_VERSION = "1.0.0"
    MODEL_PATH = "models/rental_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    
settings = Settings()