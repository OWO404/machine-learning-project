# Prédiction du Défaut de Crédit – Projet Machine Learning

## 1. Objectif du projet

Ce projet vise à prédire le défaut de paiement de cartes de crédit à partir du dataset UCI Credit Card Default.

L’objectif est double :

Construire une pipeline rigoureuse et reproductible

Comparer différents modèles dans une logique décisionnelle orientée coût

Le projet ne se limite pas à maximiser l’accuracy, mais intègre :

Gestion du déséquilibre de classes

Validation croisée stratifiée

Comparaison modèles linéaires / arborescents / deep learning

Optimisation du seuil sous contrainte de coût asymétrique

## 2. Structure du dépôt
data/raw/
    UCI_Credit_Card.csv

notebook/
    EDA.ipynb

src/
    data_loader.py
    evaluate.py
    train.py
Description

data/raw/ : données originales

notebook/EDA.ipynb : analyse exploratoire (déséquilibre, distributions, corrélations)

src/train.py : script principal d’entraînement et d’évaluation

src/evaluate.py : métriques d’évaluation

src/data_loader.py : chargement et préparation des données

## 3. Cadre expérimental

Séparation train/test stratifiée (80/20)

Validation croisée stratifiée à 5 plis

Pipeline intégrée (prétraitement + modèle)

Gestion du déséquilibre (class_weight / SMOTE)

Optimisation du seuil via fonction de coût :

Coût = c_FN × FN + c_FP × FP

avec :
c_FN = 5
c_FP = 1

## 4. Exécution

Depuis le dossier src :

python src/train.py --model gb

Modèles disponibles :

lr_base

lr_balanced

lr_smote

rf

gb

mlp

Pour lancer l’optimisation d’hyperparamètres (GB) :
python src/train.py --model gb --tune

## 5. Résultats principaux

Le Gradient Boosting obtient la meilleure PR-AUC et le coût minimal.

Les modèles arborescents surpassent la régression logistique.

Le MLP n’apporte pas d’amélioration significative par rapport au boosting.

L’optimisation du seuil améliore fortement la décision économique.

## 6. Reproductibilité

random_state fixé

Pipeline intégrée

Jeu de test isolé

Environnement contrôlé
