# app/models.py
# Defines what data the API accepts and returns

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# What the user sends to the API
class RentalRequest(BaseModel):
    """Input data for rental prediction"""
    duration_days: int = Field(..., ge=1, le=30, description="Planned rental days (1-30)")
    daily_rate: float = Field(..., ge=10, le=500, description="Cost per day in USD")
    total_cost: float = Field(..., ge=10, le=15000, description="Total rental cost")
    days_until_event: int = Field(..., ge=-10, le=30, description="Days until event (negative = started)")
    week_of_season: int = Field(..., ge=1, le=52, description="Week number in academic year")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of rental (0-23)")
    is_weekend: bool = Field(..., description="Is it weekend? (true/false)")
    is_holiday_season: bool = Field(..., description="Is holiday season? (true/false)")
    user_role: str = Field(..., description="Student, Faculty, or Staff")
    gadget_category: str = Field(..., description="Laptop, Camera, Tablet, Projector, Calculator")

# What the API returns
class PredictionResponse(BaseModel):
    """Prediction result"""
    prediction: int
    prediction_label: str
    confidence: float
    probability_short: float
    probability_long: float
    message: Optional[str] = None

# Model metrics
class MetricsResponse(BaseModel):
    """Model performance metrics"""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    model_type: str
    description: str