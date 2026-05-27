# app/main.py
# Main FastAPI application

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from .models import RentalRequest, PredictionResponse, MetricsResponse
from .ml_service import ml_service
from .config import settings

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    print("🚀 Starting ML API...")
    success = ml_service.load_models()
    if success:
        print("✅ ML models ready!")
    else:
        print("⚠️ ML models failed to load")
    yield
    # Shutdown: Clean up if needed
    print("👋 Shutting down ML API...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Predicts if a gadget rental will be short-term or long-term",
    lifespan=lifespan
)

# Enable CORS (so your AppDev frontend can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Home endpoint - API information"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "/": "This information",
            "/health": "Check API health",
            "/metrics": "Get model performance metrics",
            "/predict": "Make a prediction (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ml_service.is_loaded,
        "timestamp": time.time()
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics"""
    metrics = ml_service.get_metrics()
    return MetricsResponse(**metrics)

@app.post("/predict", response_model=PredictionResponse)
async def predict(rental: RentalRequest):
    """Predict rental duration category"""
    
    # Convert Pydantic model to dict
    rental_dict = rental.dict()
    
    # Make prediction
    result = ml_service.predict(rental_dict)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Prediction failed")
        )
    
    return PredictionResponse(
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        probability_short=result["probability_short"],
        probability_long=result["probability_long"],
        message="Prediction completed successfully"
    )

# For multiple predictions (batch)
@app.post("/predict/batch")
async def predict_batch(rentals: list[RentalRequest]):
    """Make multiple predictions at once"""
    results = []
    for rental in rentals:
        result = ml_service.predict(rental.dict())
        results.append({
            "input": rental.dict(),
            "prediction": result.get("prediction"),
            "prediction_label": result.get("prediction_label"),
            "confidence": result.get("confidence")
        })
    return {"count": len(results), "results": results}

# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)