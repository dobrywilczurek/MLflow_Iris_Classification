import mlflow
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Ładuję dane")
dane = load_iris()
X = dane.data
y = dane.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

glebokosci_do_testow = [1, 2, 3, 5, 10]

# Pętla - dla każdej głębokości z listy
for glebokosc in glebokosci_do_testow:
    print(f"Testuję głębokość: {glebokosc}")

    with mlflow.start_run(run_name=f"Drzewo_gleb_{glebokosc}"):
        model = DecisionTreeClassifier(max_depth=glebokosc)
        model.fit(X_train, y_train)

        przewidywania = model.predict(X_test)
        dokladnosc = accuracy_score(y_test, przewidywania)

        mlflow.log_param("glebokosc", glebokosc)
        mlflow.log_metric("dokladnosc", dokladnosc)

        mlflow.sklearn.log_model(model, "model")

        print(f"Dokładność: {dokladnosc:.3f}")

print("Wszystkie eksperymenty zapisane.")