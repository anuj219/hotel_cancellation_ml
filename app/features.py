import numpy as np
import pandas as pd

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12
}


def prepare_booking(booking):
    # Convert Pydantic object → dictionary
    data = booking.model_dump()

    month = MONTH_MAP[data["arrival_date_month"]]

    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    del data["arrival_date_month"]

    data["total_stays"] = (
        data["stays_in_weekend_nights"]
        + data["stays_in_week_nights"]
    )

    data["total_people"] = (
        data["adults"] + data["children"] + data["babies"]
    )

    return pd.DataFrame([data])