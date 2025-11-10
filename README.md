# Data Challenge - Apprentissage Supervisé 

Projet réalisé dans le cadre du cours **Apprentissage Supervisé Avancé** du Master M2 Mathématiques et Intelligence Artificielle à l'Université Paris-Saclay (2025).

## Table des matières

- [Challenge 1 : Classification des réservations hôtelières](#challenge-1--classification-des-réservations-hôtelières)
- [Challenge 2 : Régression de la popularité Spotify](#challenge-2--régression-de-la-popularité-spotify)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Auteurs](#auteurs)

## Challenge 1 : Classification des réservations hôtelières

### Problématique

Dans le secteur hôtelier, les annulations et les no-show entraînent des pertes financières importantes. L'objectif est de prédire le statut final d'une réservation parmi 3 catégories :

- **0** : Check-out (client s'est présenté)
- **1** : Cancel (réservation annulée)
- **2** : No-Show (client ne s'est pas présenté)


## 🎵 Challenge 2 : Régression de la popularité Spotify

### Problématique

L'industrie musicale cherche à comprendre les facteurs de popularité d'un titre. L'objectif est de prédire la popularité Spotify (0-100) à partir des caractéristiques audio et métadonnées.

## 🔧 Installation

### Prérequis

- Python 3.8+
- pip ou conda

### Installation des dépendances

```bash
# Cloner le dépôt
git clone https://github.com/mlahozy21/DATA-CHALLENGE-APPRENTISSAGE-SUPERVIS-.git
cd DATA-CHALLENGE-APPRENTISSAGE-SUPERVIS-

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
catboost>=1.0.0
joblib>=1.1.0
```

## Utilisation

### Challenge 1 : Classification

```bash
cd classification/

# 1. Entraîner les modèles de base (L0)
python train_base_models.py

# 2. Entraîner le méta-modèle (L1)
python train_meta_model.py

# 3. Générer les prédictions finales
python predict_stacking.py
```

**Sortie** : `submission.csv` contenant les prédictions pour l'ensemble de test

### Challenge 2 : Régression

```bash
cd regression/

# 1. Entraîner les modèles de base (L0)
python train_base_models.py

# 2. Entraîner le méta-modèle (L1)
python train_meta_model.py

# 3. Générer les prédictions finales
python predict_stacking.py
```

**Sortie** : `submission.csv` contenant les prédictions de popularité

### Configuration personnalisée

Les hyperparamètres et chemins peuvent être modifiés dans `config.py` :

```python
# Exemple : modifier le learning rate de LightGBM
LGBM_PARAMS = {
    'learning_rate': 0.01,  # Modifier ici
    'n_estimators': 3000,
    'num_leaves': 35,
    # ...
}
```
## Structure du projet 

DATA-CHALLENGE-APPRENTISSAGE-SUPERVIS-/
│
├── classification/
│   ├── data/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── naive_submission.csv
│   ├── models/                    # Modèles entraînés (créé automatiquement)
│   ├── processed/                 # Données pré-traitées (créé automatiquement)
│   ├── train_base_models.py      # Entraînement des modèles L0
│   ├── train_meta_model.py       # Entraînement du méta-modèle L1
│   ├── predict_stacking.py       # Génération des prédictions finales
│   ├── data_loader.py            # Chargement et nettoyage des données
│   ├── feature_engineering.py    # Création de features
│   ├── models.py                 # Définitions des modèles
│   └── config.py                 # Configuration et hyperparamètres
│
├── regression/
│   ├── data/
│   ├── models/
│   ├── processed/
│   ├── train_base_models.py
│   ├── train_meta_model.py
│   ├── predict_stacking.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── config.py
│
├── rapport.pdf                    # Rapport détaillé (méthodologie + résultats)
├── requirements.txt
└── README.md


## Auteurs

**Master 2 Mathématiques et Intelligence Artificielle - Université Paris-Saclay**

- **Marcos Lahoz**
- **Judith Le Roy**
- **Bingjian Jiang**

**Cours** : Apprentissage Supervisé et Data Challenge  
**Date** : Novembre 2025

## 📚 Références

- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Prokhorenkova et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS.
- Wolpert (1992). *Stacked Generalization*. Neural Networks, 5(2):241-259.

## License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---
