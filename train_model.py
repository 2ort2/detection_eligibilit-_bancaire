"""
train_model.py
---------------
Reproduit EXACTEMENT le pipeline de préparation de données et d'entraînement
du notebook Projet1_complete.ipynb, puis sauvegarde sur disque tout ce dont
l'API FastAPI aura besoin pour faire des prédictions cohérentes :

  - le modèle final (Random Forest optimisé par GridSearchCV)
  - la liste des colonnes utilisées à l'entraînement (pour aligner les
    nouvelles données envoyées à l'API, notamment après le one-hot encoding)
  - la médiane de Age et le mode de Embarked (pour imputer les valeurs
    manquantes de la même façon qu'à l'entraînement)

À lancer UNE FOIS, en local, avec le même Titanic-Dataset.csv que le notebook :
    python train_model.py
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# ---------------------------------------------------------------------------
# 2. Nettoyage / imputation (identique au notebook)
# ---------------------------------------------------------------------------
age_median = df["Age"].median()
embarked_mode = df["Embarked"].mode()[0]

df["Age"] = df["Age"].fillna(age_median)
df["Cabin"] = df["Cabin"].fillna(df["Cabin"].mode()[0])
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

# ---------------------------------------------------------------------------
# 3. Feature engineering : extraction du titre depuis le nom
# ---------------------------------------------------------------------------
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
     "Rev", "Sir", "Jonkheer", "Dona"],
    "Rare",
)
df["Title"] = df["Title"].replace("Mlle", "Miss")
df["Title"] = df["Title"].replace("Ms", "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

# ---------------------------------------------------------------------------
# 4. Encodage
# ---------------------------------------------------------------------------
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

# ---------------------------------------------------------------------------
# 5. Suppression des colonnes non exploitables
# ---------------------------------------------------------------------------
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# ---------------------------------------------------------------------------
# 6. X / y puis split
# ---------------------------------------------------------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 7. GridSearchCV sur Random Forest (le modèle final retenu dans le notebook)
#    -> il n'a pas besoin de mise à l'échelle, ce qui simplifie l'API.
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

final_model = grid.best_estimator_
y_pred_final = final_model.predict(X_test)

print("Meilleurs paramètres :", grid.best_params_)
print("Accuracy sur le test :", round(accuracy_score(y_test, y_pred_final), 4))

# ---------------------------------------------------------------------------
# 8. Sauvegarde des artefacts pour l'API
# ---------------------------------------------------------------------------
joblib.dump(final_model, "model.joblib")

artifacts = {
    "feature_columns": X.columns.tolist(),  # ordre + noms exacts attendus par le modèle
    "age_median": float(age_median),
    "embarked_mode": embarked_mode,
}
with open("artifacts.json", "w", encoding="utf-8") as f:
    json.dump(artifacts, f, ensure_ascii=False, indent=2)

print("\nFichiers générés : model.joblib, artifacts.json")
