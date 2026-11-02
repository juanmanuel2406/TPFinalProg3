from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv(Path(__file__).with_name("dataset.csv"))

generos = ["tango", "reggaeton", "heavy-metal", "classical", "techno"]

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

df = df[df["track_genre"].isin(generos)]
df = df.drop_duplicates(subset="track_id", keep=False)
df = df.dropna(subset=features + ["track_id", "track_genre"])

print("Canciones por genero:")
print(df["track_genre"].value_counts())

x = df[features]
y = df["track_genre"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=214159,
    stratify=y
)

model = DecisionTreeClassifier(random_state=1414213)

params = {
    "max_depth": [2, 3, 4, 5, 8, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"]
}

search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

search.fit(x_train, y_train)

model = search.best_estimator_
y_pred = model.predict(x_test)

print("Arbol de decisiones")
print("Mejores parametros:", search.best_params_)
print("Accuracy de validacion:", search.best_score_)
print("Accuracy de entrenamiento:", model.score(x_train, y_train))
print("Accuracy de prueba:", accuracy_score(y_test, y_pred))
print("Orden de los generos en la matriz:", model.classes_)
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print(classification_report(y_test, y_pred, zero_division=0))

nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred_nb = nb.predict(x_test)

print("Naive Bayes")
print("Accuracy de prueba:", accuracy_score(y_test, y_pred_nb))

cancion = x_test.iloc[[0]]
prediccion = model.predict(cancion)

print("Cancion:", df.loc[cancion.index[0], "track_name"])
print("Artista:", df.loc[cancion.index[0], "artists"])
print("Genero real:", y_test.iloc[0])
print("Genero predicho:", prediccion[0])

plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=features,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3
)
plt.title("Arbol de decisiones: primeros niveles")
plt.show()
