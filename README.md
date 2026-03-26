# Projet P2M — Smart Meter NILM (Non-Intrusive Load Monitoring)

## Vue d'ensemble

Ce projet implémente un pipeline complet de **NILM (Non-Intrusive Load Monitoring)** en Python.
L'objectif est de **désagréger** la consommation électrique totale d'un foyer (mesurée sur un compteur global) afin d'identifier et d'estimer la consommation individuelle de chaque appareil (réfrigérateur, lave-vaisselle, machine à laver, etc.) sans capteur individuel.

La méthode principale est le **Super-State Hidden Markov Model (SSHMM)** combiné à l'algorithme de **Viterbi** (classique et creux/sparse) pour l'inférence. Le pipeline complet se décompose en cinq étapes :

```
Données CSV → Pré-traitement → Détection d'événements → Clustering → Désagrégation SSHMM → Évaluation
```

---

## Architecture du projet

```
projetp2m_smart_meter/
├── PROJECT_PROMPT.txt
├── README.md
├── execution.md
├── Processed_Data_CSV/                   ← Données REFIT (à placer ici)
│   ├── House_1.csv
│   ├── House_2.csv
│   └── ... (jusqu'à House_21.csv)
├── Preprocessing/
│   └── Algorithms/
│       ├── preprocessing.py              ← Filtre Hampel + comparaison interpolation
│       └── check_preprocessing.py        ← Validation du pipeline de prétraitement
├── Event_Detection/
│   └── Algorithms/
│       ├── steady_states.py              ← Détection Hart 1985 (états stables)
│       ├── cumsum.py                     ← Algorithme CUSUM
│       ├── ca_cfar.py                    ← Algorithme CA-CFAR
│       ├── local_threshold_based.py      ← Seuillage local adaptatif
│       └── binary_segmentation.py        ← Segmentation binaire (ruptures)
├── Clustering/
│   └── Coding/
│       ├── Appliance.py                  ← Classe Appliance (clustering en ligne)
│       └── cluster.py                    ← Fonctions K-Means / MeanShift
└── HMM_Disaggregation/
    └── Super_State_HMM/
        ├── Viterbi/
        │   ├── algo_Viterbi.py           ← Viterbi classique O(K²)
        │   └── algo_SparseViterbi.py     ← Viterbi creux (optimisé)
        └── Testing/
            ├── test_Algorithm.py         ← Script principal de test et d'évaluation
            └── results/                  ← Graphiques PNG générés automatiquement
```

---

## Données (REFIT Dataset)

Le projet utilise le jeu de données **REFIT** (*Real world Energy Flexible Intelligent metering*) :
- **Source** : [REFIT Dataset – University of Strathclyde](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)
- **Fréquence d'échantillonnage** : 1 mesure toutes les 8 secondes
- **Format** : CSV par maison — colonnes `[Time, Unix, Aggregate, Appliance1, …, Appliance9]`
- **Couverture** : 20+ maisons britanniques sur plusieurs années

Placer les fichiers CSV dans `Processed_Data_CSV/` :
```
Processed_Data_CSV/
├── House_1.csv
├── House_2.csv
└── ...
```

> **Sans les données REFIT**, tous les scripts génèrent automatiquement des données synthétiques pour la démonstration.

---

## Dépendances

### Python requis : 3.8+

```bash
pip install numpy pandas matplotlib scikit-learn scipy ruptures pyarrow
```

| Package | Version minimale | Utilisation |
|---|---|---|
| numpy | ≥ 1.21 | Calculs vectoriels |
| pandas | ≥ 1.3 | Manipulation des séries temporelles |
| matplotlib | ≥ 3.4 | Visualisations |
| scikit-learn | ≥ 0.24 | KMeans, MeanShift, silhouette_score |
| scipy | ≥ 1.7 | norm, find_peaks, interpolation |
| ruptures | ≥ 1.1 | Segmentation binaire (Binseg) |
| pyarrow | optionnel | Sauvegarde Parquet |

---

## Description des modules

### 1. Pré-traitement (`Preprocessing/Algorithms/`)

#### `preprocessing.py`
- **`hampel_filter(series, window_size, n_sigmas=3)`** : Filtre de Hampel — détecte et corrige les outliers en utilisant la médiane glissante et la MAD (Median Absolute Deviation). Retourne la série filtrée et un masque booléen des outliers.
- **`compare_interpolation_methods(series, ...)`** : Compare quatre méthodes d'interpolation pandas (linéaire, polynomiale d'ordre 3 et 5, spline cubique) via MAE sur données masquées aléatoirement.

#### `check_preprocessing.py`
Script de validation qui vérifie automatiquement le bon fonctionnement du filtre Hampel et des méthodes d'interpolation.

---

### 2. Détection d'événements (`Event_Detection/Algorithms/`)

#### `steady_states.py` — Hart (1985)
Détecte les intervalles en état stable et les transitions dans un signal de puissance par balayage incrémental. Supporte 1 ou 2 mesures simultanées (puissance active ± réactive).

#### `cumsum.py` — CUSUM
- **`CUSUM_Detector`** : détecteur basé sur les sommes cumulées normalisées S_pos et S_neg.
- **`ProbCUSUM_Detector`** : variante probabiliste basée sur les p-valeurs de la loi normale.

#### `ca_cfar.py` — CA-CFAR
Détecteur adaptatif Cell-Averaging CFAR : estime le bruit local via des cellules de référence et compare chaque échantillon à un seuil adaptatif `alpha × bruit_local`.

#### `local_threshold_based.py`
Détection de pics par seuillage local adaptatif avec filtre médian, fenêtres actives et `scipy.signal.find_peaks`.

#### `binary_segmentation.py`
Segmentation binaire (`ruptures.Binseg`) + K-Means (k=2) pour identifier les cycles ON/OFF complets d'un appareil.

---

### 3. Clustering (`Clustering/Coding/`)

#### `cluster.py`
- **`cluster(X, max_num_clusters, exact_num_clusters)`** : K-Means avec sélection automatique du nombre de clusters par score de silhouette.
- **`hart85_means_shift_cluster(pair_buffer_df, columns)`** : MeanShift sur les transitions pairées de Hart.

#### `Appliance.py`
Classe `Appliance(name, series, initial_means)` — clustering en ligne, calcul des statistiques par état, construction de la matrice de transition HMM, export des paramètres (π, A, μ, Σ) en format Python.

---

### 4. Désagrégation HMM (`HMM_Disaggregation/`)

#### `Viterbi/algo_Viterbi.py`
Algorithme de Viterbi **classique** (dense, complexité O(K²)).
Interface : `disagg_algo(hmm, [y0, y1]) → (p, k, Pt1, cdone, ctotal)`

#### `Viterbi/algo_SparseViterbi.py`
Algorithme de Viterbi **creux** (sparse) : utilise des dictionnaires creuses pour sauter les transitions nulles. Économies typiques de 70–99% de calculs pour de grands modèles.
Interface identique : `disagg_algo(hmm, [y0, y1]) → (p, k, Pt1, cdone, ctotal)`

#### `Testing/test_Algorithm.py`
Script principal d'évaluation avec :
- Chargement du modèle et des données
- Validation croisée k-fold temporelle
- Calcul de l'accuracy, MAE par appareil
- Génération de graphiques PNG "Actual vs Predicted"
- Export d'un tableau de synthèse CSV

---

## Pipeline end-to-end

```
Étape 1 : Chargement CSV
          └─ parse_dates, tri chronologique

Étape 2 : Pré-traitement
          ├─ hampel_filter (suppression outliers)
          └─ interpolation (données manquantes)

Étape 3 : Détection d'événements
          └─ Hart / CUSUM / CA-CFAR / Local / Binseg

Étape 4 : Clustering
          └─ cluster() ou Appliance.updateClusters()
             → centroids, matrice de transition A, μ, σ²

Étape 5 : Construction SSHMM
          └─ Produit cartésien des états individuels
             → K_total = ∏ K_i, matrices A et B creuses

Étape 6 : Désagrégation (inférence en ligne)
          └─ algo_SparseViterbi.disagg_algo(sshmm, [y0, y1])

Étape 7 : Évaluation
          └─ MAE, F1-score, accuracy, rapport CSV
```

---

## Conventions de code

- **Langage** : Python 3.8+
- **API uniforme** pour les algorithmes de désagrégation : `disagg_algo(hmm, y) → (p, k, Pt, cdone, ctotal)`
- **Paramètres HMM** : `pi['nom']`, `a['nom']`, `mean['nom']`, `cov['nom']`
- **Graphiques** : matplotlib avec `figsize=(12, 5)` ou `(10, 5)` par défaut
- **Pandas ≥ 2.0** : `items()` utilisé à la place de `iteritems()`

---

## Points d'attention

1. **libSSHMM** : La librairie Super-State HMM de Stephen Makonin (2013–2015) n'est pas incluse. Le script `test_Algorithm.py` fournit une implémentation `ToySSHMM` de démonstration. Pour la production, implémenter `libSSHMM`, `libDataLoaders`, `libFolding` et `libAccuracy`.

2. **Fréquence d'échantillonnage** : REFIT utilise 8 secondes. Adapter `resample_rule` dans le preprocessing pour d'autres datasets.

3. **Nombre d'états par appareil** (recommandations) :
   - Réfrigérateur : 2 états (OFF, compresseur ON)
   - Lave-vaisselle : 3–5 états
   - Machine à laver : 4–6 états

4. **Split temporel recommandé** : 70% entraînement / 15% validation / 15% test par maison.

---

## Auteurs et références

- **Méthode SSHMM** : S. Makonin, F. Popowich, L. Bartram, B. Gill, I. V. Bajić (2013). *Ampds: A public smart meter dataset for load disaggregation and eco-feedback research.* IEEE EPEC.
- **Algorithme de Hart** : G. W. Hart (1985). *Prototype nonintrusive appliance load monitor.* MIT/DOE.
- **Dataset REFIT** : Murray, D. et al. (2017). *An electrical load measurements dataset of United Kingdom households from a two-year monitoring study.* Scientific Data.

---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour les détails.
