import sys

sys.path.append('..')

from main import CSVStudentTopicMatcher


def test_specialization_normalization():
    matcher = CSVStudentTopicMatcher()

    test_cases = [
        ("машинное обучение", "Machine Learning"),
        ("ML", "Machine Learning"),
        ("ml", "Machine Learning"),
        ("анализ данных", "Data Science"),
        ("data science", "Data Science"),
        ("бэкенд", "Backend"),
        ("backend", "Backend"),
        ("неизвестная специализация", "Other")
    ]

    for input_text, expected in test_cases:
        result = matcher._normalize_specialization(input_text)
        assert result == expected, f"Ошибка: '{input_text}' → '{result}' (ожидалось: '{expected}')"

    print("✅ Все тесты нормализации пройдены!")


if __name__ == "__main__":
    test_specialization_normalization()