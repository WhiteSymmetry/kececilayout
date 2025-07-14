def test_placeholder():
    assert True

def add_numbers(a, b):
    return a + b

def test_add_numbers():
    result = add_numbers(2, 3)
    assert result == 5, f"Expected 5, but got {result}"

    result2 = add_numbers(-1, 4)
    assert result2 == 3, f"Expected 3, but got {result2}"

    result3 = add_numbers(0, 0)
    assert result3 == 0, f"Expected 0, but got {result3}"

test_add_numbers()
