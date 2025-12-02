import pandas as pd
import numpy as np
import joblib
import sys
from datetime import datetime
from sklearn.linear_model import LogisticRegression


class FeedbackTrainer:
    def __init__(self):
        self.feedback_file = 'feedback.csv'
        self.model_file = 'feedback_model.pkl'

    def collect_feedback(self, results_file='matches.csv'):
        """Собрать оценки для результатов"""
        try:
            results = pd.read_csv(results_file)
        except:
            print(f"❌ Файл {results_file} не найден")
            return

        print(f"\n📊 Оцените подборы из {results_file}")
        print("=" * 50)

        for i, row in results.head(20).iterrows():  # первые 20
            print(f"\n#{i + 1} Студент: {row.get('student_name', '?')}")
            print(f"   Тема: {row.get('theme_title', '?')}")
            print(f"   Оценка системы: {row.get('total_score', 0):.2f}")

            while True:
                rating = input("   Хороший подбор? (y=да/n=нет/s=пропустить): ").lower()
                if rating in ['y', 'n', 's']:
                    break

            if rating != 's':
                feedback = pd.DataFrame([{
                    'student_id': row.get('student_id', ''),
                    'theme_id': row.get('theme_id', ''),
                    'system_score': row.get('total_score', 0),
                    'human_score': 1 if rating == 'y' else 0,
                    'date': datetime.now().strftime('%Y-%m-%d')
                }])

                # Сохраняем
                feedback.to_csv(self.feedback_file, mode='a',
                                header=not pd.io.common.file_exists(self.feedback_file),
                                index=False)
                print(f"   ✅ Оценка {'хорошо' if rating == 'y' else 'плохо'} сохранена")

        print(f"\n✅ Оценки сохранены в {self.feedback_file}")

    def train_model(self):
        """Обучить простую модель на собранных оценках"""
        try:
            feedback = pd.read_csv(self.feedback_file)
        except:
            print("❌ Нет файла с оценками")
            return

        if len(feedback) < 10:
            print(f"❌ Нужно минимум 10 оценок, есть {len(feedback)}")
            return

        print(f"🧠 Обучение модели на {len(feedback)} оценках...")

        # Простые признаки (можно расширить)
        X = feedback[['system_score']].values
        y = feedback['human_score'].values

        # Простая модель
        model = LogisticRegression()
        model.fit(X, y)

        # Сохраняем
        joblib.dump(model, self.model_file)

        accuracy = model.score(X, y)
        print(f"✅ Модель обучена! Точность: {accuracy:.1%}")
        print(f"📁 Сохранена: {self.model_file}")

        # Статистика
        good = sum(y == 1)
        bad = sum(y == 0)
        print(f"📊 Оценок: 👍 {good} | 👎 {bad}")

    def use_trained_model(self, system_score):
        """Использовать обученную модель для коррекции оценки"""
        try:
            model = joblib.load(self.model_file)
        except:
            print("ℹ️ Модель еще не обучена, использую базовую оценку")
            return system_score

        # Корректируем оценку
        prob_good = model.predict_proba([[system_score]])[0][1]
        adjusted_score = 0.7 * system_score + 0.3 * prob_good

        return round(adjusted_score, 3)

    def show_stats(self):
        """Показать статистику"""
        try:
            feedback = pd.read_csv(self.feedback_file)
            print(f"\n📈 Статистика:")
            print(f"   Всего оценок: {len(feedback)}")
            print(f"   Хороших: {sum(feedback['human_score'] == 1)}")
            print(f"   Плохих: {sum(feedback['human_score'] == 0)}")

            if 'system_score' in feedback.columns:
                avg_system = feedback['system_score'].mean()
                avg_human = feedback['human_score'].mean()
                print(f"   Средняя оценка системы: {avg_system:.2f}")
                print(f"   Средняя оценка людей: {avg_human:.2f}")

        except:
            print("❌ Нет данных для статистики")


# 📌 Использование в основном коде
def enhance_with_feedback(base_score, student_data=None):
    """
    Функция для вставки в main.py
    Использование:
        score = calculate_comprehensive_score(...)  # обычная оценка
        enhanced_score = enhance_with_feedback(score)  # с учетом обучения
    """
    trainer = FeedbackTrainer()
    return trainer.use_trained_model(base_score)


# 🚀 Запуск из командной строки
if __name__ == "__main__":
    trainer = FeedbackTrainer()

    if len(sys.argv) < 2:
        print("Использование:")
        print("  python train_feedback.py collect [matches.csv]  # собрать оценки")
        print("  python train_feedback.py train                  # обучить модель")
        print("  python train_feedback.py stats                  # статистика")
        return

    command = sys.argv[1]

    if command == "collect":
        file = sys.argv[2] if len(sys.argv) > 2 else "matches.csv"
        trainer.collect_feedback(file)

    elif command == "train":
        trainer.train_model()

    elif command == "stats":
        trainer.show_stats()

    elif command == "test":
        if len(sys.argv) > 2:
            score = float(sys.argv[2])
            enhanced = trainer.use_trained_model(score)
            print(f"Базовая оценка: {score}")
            print(f"С учетом обучения: {enhanced}")

    else:
        print(f"❌ Неизвестная команда: {command}")