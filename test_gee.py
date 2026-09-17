# Linear Regression using sklearn
# Source: geeksforgeeks.org
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Генерация синтетических данных
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # признак
y = 2.5 * X.squeeze() + 1.5 + np.random.randn(100) * 2  # целевая переменная с шумом

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Предсказание
y_pred = model.predict(X)

# Вывод коэффициентов
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Data points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
