from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1.3 Category codling (convert strings to numbers)
num_transformer = StandardScaler()  # Standardize numerical features
cat_transformer = OneHotEncoder(handle_unknown='ignore')  # One-hot encode categorical features

# 1.2 Normalisation


def create_pipeline(preprocessor):
    # pipeline = Pipeline([
    #     ('preprocessor', preprocessor),  # Data transformation
    #     ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ML model
    # ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),                 # масштабуємо дані
        ("model", RandomForestClassifier())            # модель
    ])
    return pipeline

# 4. Hyperparameter tuning
            # бере різні набори налаштувань,
            # навчає модель з кожним набором,
            # перевіряє, яка комбінація дає найкращі результати,
            # вибирає найкращу.
    # 4.1 GridSearchCV Перебираємо всі можливі варіанти (як перепробувати всі рецепти).
def create_GridSearchCV(pipeline):

    # param_grid = {
    #     'model__n_estimators': [100, 200],
    #     'model__max_depth': [3, 5, 7],
    #     'model__min_samples_split': [5, 10, 20]
    # }

    param_grid = [

        # Random Forest
        {
            'model': [RandomForestClassifier()],
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7]
        },

        # Logistic Regression
        {
            'model': [LogisticRegression(max_iter=200)],
            'model__C': [0.1, 1.0, 3.0],
            'model__solver': ['lbfgs']
        },

        # Support Vector Machine
        {
            'model': [SVC(probability=True)],
            'model__C': [0.5, 1, 3],
            'model__kernel': ['rbf', 'linear']
        },

        # Decision Tree
        {
            'model': [DecisionTreeClassifier()],
            'model__max_depth': [3, 5, 7, 10]
        }
    ]

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    return grid

