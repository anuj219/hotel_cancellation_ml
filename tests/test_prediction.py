def test_prediction(client, valid_booking):
    response = client.post("/predict", json=valid_booking)

    assert response.status_code == 200  

    body = response.json()

    assert 0 <= body["cancel_probability"] <= 1
    assert body["prediction"] == 1
    assert body["result"] == "Likely to cancel"
    assert body["threshold"] == 0.3999


def test_negative_lead_time_rejected(client, valid_booking):
    valid_booking["lead_time"] = -5

    response = client.post("/predict", json=valid_booking)

    assert response.status_code == 422


def test_invalid_hotel_rejected(client, valid_booking):
    valid_booking["hotel"] = "Unknown Hotel"

    response = client.post("/predict", json=valid_booking)

    assert response.status_code == 422