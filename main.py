"""
main.py
-------
API FastAPI qui sert le modèle Titanic entraîné par train_model.py.

Lancement local :
    uvicorn main:app --reload

Documentation interactive auto-générée :
    http://127.0.0.1:8000/docs
"""

import json
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Chargement du modèle et des artefacts UNE SEULE FOIS au démarrage de l'API
# (pas à chaque requête, ce serait beaucoup trop lent)
# ---------------------------------------------------------------------------
model = joblib.load("model.joblib")

with open("artifacts.json", "r", encoding="utf-8") as f:
    artifacts = json.load(f)

FEATURE_COLUMNS = artifacts["feature_columns"]
AGE_MEDIAN = artifacts["age_median"]
EMBARKED_MODE = artifacts["embarked_mode"]

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Prédit la probabilité de survie d'un passager du Titanic.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schéma d'entrée : ce que l'utilisateur de l'API doit envoyer.
# Ce sont les champs "bruts" du dataset original (avant tout traitement).
# Pydantic valide automatiquement les types et renvoie une erreur 422 claire
# si un champ est manquant ou mal typé.
# ---------------------------------------------------------------------------
class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Classe (1, 2 ou 3)")
    Name: str = Field(..., description="Nom complet, ex: 'Braund, Mr. Owen Harris'")
    Sex: str = Field(..., description="'male' ou 'female'")
    Age: Optional[float] = Field(None, description="Âge (laisser vide si inconnu)")
    SibSp: int = Field(0, description="Nb de frères/soeurs/époux à bord")
    Parch: int = Field(0, description="Nb de parents/enfants à bord")
    Fare: float = Field(..., description="Prix du billet")
    Embarked: Optional[str] = Field(None, description="'C', 'Q' ou 'S'")

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 3,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 22,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
            }
        }


class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float


# ---------------------------------------------------------------------------
# Reproduit EXACTEMENT le feature engineering du notebook / de train_model.py,
# mais pour une seule ligne (un seul passager) reçue par l'API.
# ---------------------------------------------------------------------------
def preprocess(passenger: Passenger) -> pd.DataFrame:
    df = pd.DataFrame([passenger.model_dump()])

    # Imputation avec les valeurs figées à l'entraînement (jamais recalculées ici)
    df["Age"] = df["Age"].fillna(AGE_MEDIAN)
    df["Embarked"] = df["Embarked"].fillna(EMBARKED_MODE)

    # Titre extrait du nom
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
         "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare",
    )
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
    df["Title"] = df["Title"].fillna("Rare")  # sécurité si le nom ne matche aucun titre

    # Encodage
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=False)

    df = df.drop(["Name"], axis=1)

    # Alignement STRICT sur les colonnes vues à l'entraînement :
    # - toute colonne manquante (ex: Title_Rare absent sur ce passager) -> 0
    # - toute colonne en trop (catégorie jamais vue à l'entraînement) -> supprimée
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df


@app.get("/")
def root():
    return {"message": "API opérationnelle. Voir /docs pour tester le modèle."}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: Passenger):
    X = preprocess(passenger)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])  # proba de la classe "survécu"

    return PredictionResponse(
        survived=prediction,
        survival_probability=round(probability, 4),
    )
