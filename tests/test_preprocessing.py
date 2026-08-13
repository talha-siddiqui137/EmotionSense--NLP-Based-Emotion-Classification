from src.preprocessing import preprocess_text


def test_lowercases_text():
    assert preprocess_text("HELLO") == "hello"


def test_removes_punctuation():
    result = preprocess_text("Hello!!! World???")
    assert "!" not in result
    assert "?" not in result


def test_removes_numbers():
    result = preprocess_text("I have 123 apples")
    assert "123" not in result


def test_removes_stopwords():
    result = preprocess_text("I am the happy one")
    assert "am" not in result
    assert "the" not in result


def test_empty_string_returns_empty():
    assert preprocess_text("") == ""