from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier



# 1.2 Normalisation
# 1.3 Category codling (convert strings to numbers)
def create_pipeline():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),   # заповнюємо пропуски
        ("scaler", StandardScaler()),                  # масштабуємо дані
        ("model", RandomForestClassifier())            # модель
    ])
    return pipeline