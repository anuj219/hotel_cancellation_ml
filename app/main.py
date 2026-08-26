import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal  # for input validation

from pathlib import Path
# -------------------

app = FastAPI()



# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
# __file__: A built-in Python variable pointing to the path of the current script
# resolve(): Finds the full, absolute path
model_path = BASE_DIR / "model" / "hotel_cancellation_pipeline.pkl"


model = joblib.load(
    # r"C:\Users\anujv\Desktop\Programming\Codes\Any Notes\ML\hotel_cancellation_ml\model\hotel_cancellation_pipeline.pkl"
    model_path
)


THRESHOLD = 0.3999


# ------------------------------------------------------------
# Month mapping
# ------------------------------------------------------------

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class Booking(BaseModel):

    hotel: Literal["City Hotel", "Resort Hotel"]

    lead_time: int = Field(ge=0)

    arrival_date_month: Literal[
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)

    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)

    adults: int = Field(ge=0)
    children: int = Field(ge=0)
    babies: int = Field(ge=0)

    meal: Literal['BB', 'FB', 'HB', 'SC']

    market_segment: str
    distribution_channel: str

    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)

    booking_changes: int = Field(ge=0)
    days_in_waiting_list: int = Field(ge=0)

    adr: float

    required_car_parking_spaces: int = Field(ge=0)

    deposit_type: str
    customer_type: str


# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------

@app.post("/predict")
def predict(booking: Booking):

    # Convert Pydantic object → dictionary
    data = booking.model_dump()

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    month = MONTH_MAP[data["arrival_date_month"]]

    data["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    del data["arrival_date_month"]

    data["total_stays"] = (
        data["stays_in_weekend_nights"]
        + data["stays_in_week_nights"]
    )

    data["total_people"] = (
        data["adults"]
        + data["children"]
        + data["babies"]
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------
    booking_df = pd.DataFrame([data])

    probability = model.predict_proba(booking_df)[0, 1]

    prediction = int(probability >= THRESHOLD)

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "cancel_probability": float(probability),
        "prediction": prediction,
        "result": (
            "Likely to cancel"
            if prediction == 1
            else "Likely not to cancel"
        ),
        "threshold": THRESHOLD
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }