#import sns

import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_models(grid, X_test, y_test):
    results = []

    for i, params in enumerate(grid.cv_results_['params']):
        model_name = type(params["model"]).__name__
        cv_score = grid.cv_results_['mean_test_score'][i]

        # Беремо модель з конкретного run
        model = grid.best_estimator_ if params == grid.best_params_ else None

        # Тестовий скор (лише для best estimator)
        if model:
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
        else:
            test_acc = None

        results.append({
            "Model": model_name,
            "CV Score": cv_score,
            "Test Accuracy (best only)": test_acc,
            "Params": params
        })

    df = pd.DataFrame(results)
    return df