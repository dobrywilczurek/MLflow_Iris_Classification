import mlflow
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "b42fceaa742a4bce899ef4e69f65e65c"

print("Wczytuję model")
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
print(RUN_ID)

przyklady = [
    [5.1, 3.5, 1.4, 0.2],  # to powinien być setosa (klasa 0)
    [6.0, 3.0, 4.8, 1.8],  # to powinien być virginica (klasa 2)
    [5.5, 2.4, 3.8, 1.1],  # to powinien być versicolor (klasa 1)
]

for i, probka in enumerate(przyklady):
    probka = np.array(probka).reshape(1, -1)

    klasa = model.predict(probka)[0]

    print(f"Próbka {i + 1}: {probka}")
    print(f"Przewidziana klasa: {klasa}")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(probka)[0]
        print(f"Pewność: {max(prob):.2f}")

    print()