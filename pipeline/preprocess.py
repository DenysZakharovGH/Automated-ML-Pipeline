from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def create_pipeline():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),   # заповнюємо пропуски
        ("scaler", StandardScaler()),                  # масштабуємо дані
        ("model", RandomForestClassifier())            # модель
    ])
    return pipeline