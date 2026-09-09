# tests your feature engineering independently of FastAPI and the Random Forest.
# protects against training-serving skew: 
#   when the transformations used during inference differ from those used during training

import pytest

from app.features import prepare_booking
from app.main import Booking


def test_feature_engineering(valid_booking):
    booking = Booking(**valid_booking)

    X = prepare_booking(booking)

    assert X["total_stays"].iloc[0] == 5  # 5, cause, we know the Valid_booking data has this
    # Assert - expect this condition to be true. If it isn't, fail the test.
    assert X["total_people"].iloc[0] == 3

    assert "arrival_date_month" not in X.columns
    assert "month_sin" in X.columns
    assert "month_cos" in X.columns