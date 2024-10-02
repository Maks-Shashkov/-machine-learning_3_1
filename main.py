import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Шаг 1: Загрузка данных
data = pd.read_csv('winequality-red.csv')

# Шаг 2: Преобразование качества вина в классы: "Хорошее" (1) и "Не качественное" (0)
data['class'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Убираем столбец 'quality', он нам не нужен для классификации
X = data.drop(columns=['quality', 'class'])
y = data['class']

# Шаг 3: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Шаг 4: Балансировка классов с помощью SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Обучение модели логистической регрессии с классами 'balanced'
model = LogisticRegression(C=1, solver='liblinear', class_weight='balanced')
model.fit(X_resampled, y_resampled)

# Шаг 5: Получение коэффициентов модели
coefficients = model.coef_[0]  # Коэффициенты при параметрах
intercept = model.intercept_[0]  # Свободный член

# Распечатка коэффициентов для анализа
print("Коэффициенты модели:")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef}")

# Применим коэффициенты к характеристикам одного "Хорошего" и одного "Плохого" вина
bad_wine = X_test_scaled[0]
bad_wine_score = np.dot(bad_wine, coefficients) + intercept

# Пример "Хорошего" вина (найдем в тестовой выборке)
good_wine_index = y_test[y_test == 1].index[0]  # первый индекс, где вино хорошее
good_wine = X_test_scaled[list(X_test.index).index(good_wine_index)]
good_wine_score = np.dot(good_wine, coefficients) + intercept

print(f"Счет для 'Плохого' вина: {bad_wine_score}")
print(f"Счет для 'Хорошего' вина: {good_wine_score}")

# Проверим, что значения имеют разные знаки
if bad_wine_score * good_wine_score < 0:
    print("Модель правильно различает классы!")
else:
    print("Модель ошибается в различении классов.")

# Шаг 6: Рассчитаем метрики Accuracy, Precision и Recall на тестовых данных
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")






