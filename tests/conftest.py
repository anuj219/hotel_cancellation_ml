import pytest
import numpy as np

from fastapi.testclient import TestClient
import app.main as main


class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.60, 0.40]] * len(X))


@pytest.fixture
def client(monkeypatch):
    # monkeypatch -> pytest tool that temporarily replaces something during a test.
    monkeypatch.setattr(
        main.joblib,
        "load",
        lambda path: FakeModel() #small anonymous function
    )

    with TestClient(main.app) as test_client:  #creates a test client for our 'app'
        # helps make requests without opening a browser or starting Uvicorn on port 8000
        yield test_client # yeild gives the client to the test while keeping the fixture active


# called mocking
# We're testing whether the API behaves correctly when a model returns probabilities



# every test needs a valid booking
# You don't want to copy that entire dictionary into all 20 tests.
@pytest.fixture  # --> fixture: reusable piece of setup that pytest provides to tests
def valid_booking():
    return {
        "hotel": "City Hotel",
        "lead_time": 100,
        "arrival_date_month": "August",
        "arrival_date_week_number": 32,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 1,
        "babies": 0,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "days_in_waiting_list": 0,
        "adr": 120.0,
        "required_car_parking_spaces": 0,
        "deposit_type": "No Deposit",
        "customer_type": "Transient"
    }