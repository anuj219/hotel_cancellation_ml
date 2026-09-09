# actual saved pipeline can load and predict?
import os
import joblib
import pytest

from app.main import model_path, Booking
from app.features import prepare_booking


@pytest.mark.integration
def test_real_model_prediction(valid_booking):
    if not model_path.exists():
        pytest.skip("Real model artifact is not available")

    model = joblib.load(model_path)

    booking = Booking(**valid_booking)
    X = prepare_booking(booking)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0, 1] <= 1