from sklearn.neural_network import MLPClassifier
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline as SkPipeline  # Pour les sous-pipelines

from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline compatible SMOTE
from imblearn.over_sampling import SMOTE

from data_loader import load_credit_data, make_split
from evaluate import (
    evaluate_binary_classifier,
    pretty_print_metrics,
    find_optimal_threshold,
)

RANDOM_STATE = 4242


def build_preprocessor(X, use_scaler: bool):
    """
    Construit le préprocesseur (ColumnTransformer) :
    - Numérique : imputation médiane (+ éventuellement standardisation)
    - Catégoriel : imputation + OneHot (sortie dense pour compatibilité SMOTE)
    """
    cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipeline numérique
    if use_scaler:
        numeric_pipe = SkPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipe = SkPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

    # Pipeline catégoriel (sortie dense)
    categorical_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0  # Force une sortie dense globale
    )

    return preproc


def build_model(model_name: str):
    """
    Construit le modèle classique selon l'option choisie.
    """
    if model_name == "lr_base":
        return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)

    if model_name == "lr_balanced":
        return LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE,
            class_weight="balanced"
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    if model_name == "gb":
        return GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )



    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,                 # régularisation L2
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,        # arrêt anticipé via validation interne
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE
        )
    raise ValueError(f"Modèle inconnu: {model_name}")
    
    def build_pipeline(X_train, model_name: str):
    """
    Construit une pipeline complète.
    SMOTE est activé uniquement pour l'option lr_smote.
    """
    # Scaling utile pour LR / MLP ; inutile pour RF/GB
    use_scaler = model_name in {"lr_base", "lr_balanced", "lr_smote", "mlp"}
    preproc = build_preprocessor(X_train, use_scaler=use_scaler)

    if model_name == "lr_smote":
        model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
        pipe = ImbPipeline(steps=[
            ("preprocess", preproc),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model)
        ])
        return pipe

    # Autres modèles : pas de SMOTE ici (comparaison classique)
    model = build_model(model_name)
    pipe = ImbPipeline(steps=[
        ("preprocess", preproc),
        ("smote", "passthrough"),
        ("model", model)
    ])
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/UCI_Credit_Card.csv")
    parser.add_argument("--model", type=str, default="rf",
                        choices=["lr_base", "lr_balanced", "lr_smote", "rf", "gb","mlp"])
    parser.add_argument("--c_fn", type=float, default=5.0)
    parser.add_argument("--c_fp", type=float, default=1.0)
    args = parser.parse_args()

    X, y = load_credit_data(args.csv)

    ratio = y.mean()
    print(f"Lignes        : {len(y)}")
    print(f"Variables     : {X.shape[1]}")
    print(f"Taux de défaut: {ratio:.4%}")
    print(f"Modèle        : {args.model}")

    X_train, X_test, y_train, y_test = make_split(X, y, test_size=0.2)

    pipe = build_pipeline(X_train, model_name=args.model)
    pipe.fit(X_train, y_train)

    # Probabilités classe positive (défaut)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Évaluation standard (seuil 0.5)
    metrics = evaluate_binary_classifier(y_test.to_numpy(), y_proba, threshold=0.5)
    pretty_print_metrics(metrics)

    # Optimisation du seuil sous coût asymétrique
    best = find_optimal_threshold(
        y_test.to_numpy(), y_proba,
        c_fn=args.c_fn, c_fp=args.c_fp,
        t_min=0.05, t_max=0.95, n_grid=181
    )

    print("\n=== Optimisation du seuil (Coût asymétrique) ===")
    print(f"c_FN={args.c_fn:.1f}, c_FP={args.c_fp:.1f}")
    print(f"Meilleur seuil : {best['threshold']:.3f}")
    print(f"Coût minimal   : {best['cost']:.1f}")
    print("Matrice de confusion au seuil optimal [[TN, FP],[FN, TP]] :")
    print(best["cm"])


if __name__ == "__main__":
    main()
