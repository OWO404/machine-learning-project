import argparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_credit_data, make_split
from evaluate import evaluate_binary_classifier, pretty_print_metrics

RANDOM_STATE = 42


def build_layer0_lr_pipeline(X):
    """
    Construction du pipeline baseline (Layer 0).
    - Imputation des valeurs manquantes
    - Encodage des variables catégorielles
    - Standardisation des variables numériques
    - Régression Logistique sans gestion du déséquilibre
    """

    # Colonnes catégorielles pertinentes dans ce dataset
    cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipeline numérique : imputation médiane + standardisation
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Pipeline catégoriel : imputation + One-Hot Encoding
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Assemblage via ColumnTransformer
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    # Modèle baseline : régression logistique classique
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("model", model)
    ])

    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/content/UCI_Credit_Card.csv")
    args, unknown = parser.parse_known_args()

    # Chargement des données
    X, y = load_credit_data(args.csv)

    # Affichage du déséquilibre de classes
    ratio = y.mean()
    print(f"Lignes        : {len(y)}")
    print(f"Variables     : {X.shape[1]}")
    print(f"Taux de défaut: {ratio:.4%}")

    # Split stratifié
    X_train, X_test, y_train, y_test = make_split(X, y, test_size=0.2)

    # Construction et entraînement du pipeline
    pipe = build_layer0_lr_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # Probabilités de la classe positive (défaut)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Évaluation baseline avec seuil 0.5
    metrics = evaluate_binary_classifier(y_test.to_numpy(), y_proba, threshold=0.5)
    pretty_print_metrics(metrics)


if __name__ == "__main__":
    main()
