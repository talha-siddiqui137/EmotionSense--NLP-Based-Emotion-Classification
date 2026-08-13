import pytest

from src.predict import predict_emotion


def test_predict_returns_a_string():
    result = predict_emotion("I am feeling happy today")
    assert isinstance(result, str)


def test_predict_returns_known_emotion():
    valid_emotions = {"anger", "fear", "joy", "love", "sadness", "surprise"}
    result = predict_emotion("I am feeling happy today")
    assert result in valid_emotions


def test_empty_input_raises_error():
    with pytest.raises(ValueError):
        predict_emotion("")