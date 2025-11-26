# what is Pipeline really is
# what is a difference with templates
#
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.preprocess import create_pipeline
from utils.path_keeper import path_to_csv

# Special features
# 1. MLflow



# 1. Preprocessing
    # 1.1 deal with None
    # 1.2 Normalisation
    # 1.3 Category codling (convert strings to numbers)
# 2. Feature engineering
    # 2.1 Add new feature
    # 2.1 Delete old feature
# 3. Model Selection
    # 3.1 Choose between paar model to taste
# 4. Hyperparameter tuning
            # бере різні набори налаштувань,
            # навчає модель з кожним набором,
            # перевіряє, яка комбінація дає найкращі результати,
            # вибирає найкращу.
    # 4.1 GridSearchCV Перебираємо всі можливі варіанти (як перепробувати всі рецепти).
    # 4.2 RandomizedSearchCV Випадково пробуємо різні комбінації.
# 5. Evaluation
            # Для класифікації (так/ні):
            # accuracy — відсоток правильних відповідей
            # precision / recall
            # F1 score
            # confusion matrix
            # Для задач з числами (регресія):
            # MAE — середня помилка
            # MSE — квадрат помилки
            # RMSE
            # R²
    # 5.1 Metrics
    # 5.2 graf
# 6. Save + load pipeline
    # 6.1 joblib
    # 6.2 pickle  дозволяє зберегти Python-об’єкт у файл та завантажити його знову.

# 1. first tast to test it with Model I wrote before alone
# titanic case

# 1. Preprocessing
csv_data = pd.read_csv(path_to_csv)

# 1.1 deal with None
csv_data = csv_data.dropna()

columns_lst = csv_data.columns.to_list()

features = columns_lst[:-1]
target = columns_lst[-1]

# Define target and features
X = csv_data[features]
y = csv_data[target]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

pipeline = create_pipeline()

pipeline.fit(X_train, y_train)
print("Model training complete!")
