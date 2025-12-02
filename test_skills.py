import sys

sys.path.append('..')

from main import CSVStudentTopicMatcher


def test_skill_extraction():
    """Тест извлечения навыков из текста"""
    matcher = CSVStudentTopicMatcher()

    test_cases = [
        ("Знаю Python и Docker", ["python", "docker"]),
        ("Работал с React и JavaScript", ["javascript"]),
        ("Опыт в машинном обучении и нейросетях", ["ml"]),
        ("SQL и базы данных", ["sql"]),
        ("", []),
    ]

    for input_text, expected in test_cases:
        result = matcher._extract_skills(input_text)
        assert sorted(result) == sorted(expected), f"Ошибка: '{input_text}' → {result} (ожидалось: {expected})"

    print("✅ Все тесты извлечения навыков пройдены!")


if __name__ == "__main__":
    test_skill_extraction()