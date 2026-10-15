# apna-demo
this is my first repository
by sahilbhaii


"""Simple ML demo: Iris classification with scikit-learn."""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    # 1. Load data
    data = load_iris()
    X, y = data.data, data.target

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=data.target_names))

    # 5. Save model
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")


if __name__ == "__main__":
    main()
