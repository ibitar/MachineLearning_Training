import io
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(path: str):
    """Load the wine quality dataset handling the split header."""
    with open(path) as f:
        lines = f.read().splitlines()
    header_line = lines[0] + lines[1]
    data_lines = lines[2:]
    content = '\n'.join([header_line] + data_lines)
    return pd.read_csv(io.StringIO(content), sep=';')


def main():
    df = load_data('data/winequality-red.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Baseline model without hyperparameter tuning
    pipeline.fit(X_train, y_train)
    baseline_pred = pipeline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")

    # Grid search hyperparameter tuning
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print(f"Best cross-val accuracy: {grid_search.best_score_:.3f}")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Accuracy with best params: {test_accuracy:.3f}")
    print(f"Accuracy improvement: {test_accuracy - baseline_accuracy:.3f}")


if __name__ == '__main__':
    main()
