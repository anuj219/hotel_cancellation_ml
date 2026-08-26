import joblib
import numpy as np
import pandas as pd

model = joblib.load(
    r"C:\Users\anujv\Desktop\Programming\Codes\Any Notes\ML\hotel_cancellation_ml\notebooks\hotel_cancellation_pipeline.pkl"
)

month_map = {
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
# cause, month encoding was done before pipeline and thus, has to be done seperatetly here

# booking = {
#     "hotel": "Resort Hotel",
#     "lead_time": 0,
#     "arrival_date_month": 'July',
#     "arrival_date_week_number": 27,
#     "arrival_date_day_of_month": 1,
#     "stays_in_weekend_nights": 0,
#     "stays_in_week_nights": 0,
#     "adults": 2,
#     "children": 0,
#     "babies": 0,
#     "meal": "BB",
#     "market_segment": "Direct",
#     "distribution_channel": "Direct",
#     "is_repeated_guest": 0,
#     "previous_cancellations": 0,
#     "previous_bookings_not_canceled": 0,
#     "booking_changes": 4,
#     "deposit_type": "No Deposit",
#     "days_in_waiting_list": 0,
#     "adr": 100,
#     "required_car_parking_spaces": 0,
#     "customer_type": "Transient"
# }
# booking = {
#     "hotel": "Resort Hotel",
#     "lead_time": 85,
#     "arrival_date_month": 'July',
#     "arrival_date_week_number": 27,
#     "arrival_date_day_of_month": 1,
#     "stays_in_weekend_nights": 0,
#     "stays_in_week_nights": 3,
#     "adults": 2,
#     "children": 0,
#     "babies": 0,
#     "meal": "BB",
#     "market_segment": "Online TA",
#     "distribution_channel": "TS/TO",
#     "is_repeated_guest": 0,
#     "previous_cancellations": 0,
#     "previous_bookings_not_canceled": 0,
#     "booking_changes": 0,
#     "deposit_type": "No Deposit",
#     "days_in_waiting_list": 0,
#     "adr": 82,
#     "required_car_parking_spaces": 0,
#     "customer_type": "Transient"
# }
booking = {
  "hotel": "Resort Hotel",
  "lead_time": 85,
  "arrival_date_month": "July",
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 1,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 3,
  "adults": 2,
  "children": 0,
  "babies": 0,
  "meal": "BB",
  "market_segment": "Online TA",
  "distribution_channel": "TA/TO",
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "booking_changes": 0,
  "days_in_waiting_list": 0,
  "adr": 82,
  "required_car_parking_spaces": 0,
  "deposit_type": "No Deposit",
  "customer_type": "Transient"
}

def predict_cancellation(booking):
    booking = booking.copy()

    #-----------
    # Cyclic Encoding
    month_name = booking.pop("arrival_date_month")
    if month_name not in month_map:
        raise ValueError(
            f"Invalid month: {month_map}. "
            f"Expected one of: {list(month_map.keys())}"
        )
    month = month_map[month_name]

    booking["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )
    booking["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    #-----------
    # feature engineering
    booking["total_stays"] = (
        booking["stays_in_weekend_nights"]
        + booking["stays_in_week_nights"]
    )

    booking["total_people"] = (
        booking["adults"]
        + booking["children"]
        + booking["babies"]
    )


    booking_df = pd.DataFrame([booking])

    probability = model.predict_proba(booking_df)[0, 1]

    prediction = int(probability >= 0.3999)

    return {
        "cancel_probability": probability,
        "prediction": prediction,
        "result": (
            "Likely to cancel"
            if prediction == 1
            else "Likely not to cancel"
        )
    }  

result = predict_cancellation(booking)
print(result)


# When running from another file
if __name__ == "__main__":

    result = predict_cancellation(booking)

    print("\nHotel Cancellation Prediction")
    print("-----------------------------")
    print(f"Cancellation probability : {result['cancel_probability']:.4f}")
    print(f"Prediction               : {result['result']}")
    print(f"Threshold used           : {0.3999}")