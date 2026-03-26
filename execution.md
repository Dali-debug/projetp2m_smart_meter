# Guide d'exécution — Projet NILM Smart Meter

Ce document explique comment exécuter chaque composant du pipeline NILM, les arguments attendus, et le résultat produit pour chaque exécution.

---

## Prérequis

### 1. Installation des dépendances

```bash
pip install numpy pandas matplotlib scikit-learn scipy ruptures pyarrow
```

### 2. Données (optionnel)

Téléchargez le jeu de données REFIT depuis :
https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

Placez les fichiers CSV dans :
```
Processed_Data_CSV/
├── House_1.csv
├── House_2.csv
└── ...
```

> **Sans les données REFIT**, tous les scripts génèrent automatiquement des **données synthétiques** pour la démonstration. Aucune erreur ne sera levée.

---

## Structure des commandes

Tous les scripts sont exécutés depuis la **racine du projet** :
```bash
cd /chemin/vers/projetp2m_smart_meter
```

---

## Étape 1 — Pré-traitement

### `Preprocessing/Algorithms/preprocessing.py`

**Rôle** : Démontre le filtre de Hampel et la comparaison des méthodes d'interpolation.

```bash
python Preprocessing/Algorithms/preprocessing.py
```

**Arguments optionnels** : aucun (le chemin vers `House_1.csv` est détecté automatiquement).

**Résultat attendu** :
```
[Hampel] Outliers detected: 71 / 2880
```
- **Figure 1** : Série temporelle agrégée avec les outliers en rouge et une vue zoomée en inset.
- **Figure 2** : Comparaison visuelle des quatre méthodes d'interpolation sur un segment de 200 points.
- **Console** :
  ```
  === Interpolation MAE results (lower is better) ===
    linear       MAE = 13.6458 W
    spline3      MAE = 24.6390 W
    poly3        MAE = 25.1516 W
    poly5        MAE = 32.7805 W
    zero_fill    MAE = 318.9477 W
  ```

---

### `Preprocessing/Algorithms/check_preprocessing.py`

**Rôle** : Script de validation qui vérifie automatiquement le bon fonctionnement des fonctions de prétraitement.

```bash
python Preprocessing/Algorithms/check_preprocessing.py
# ou avec un fichier CSV personnalisé :
python Preprocessing/Algorithms/check_preprocessing.py Processed_Data_CSV/House_1.csv
```

**Résultat attendu** :
```
============================================================
CHECK 1 — Hampel filter
============================================================
  Input  samples  : 2,880
  Outliers found  : 71  (2.465 %)
  Mean shift      : +1.1231 W
  Std  shift      : +46.5029 W
  [OK] All sanity checks passed.

============================================================
CHECK 2 — Interpolation methods
============================================================
  [OK] All interpolation methods returned valid MAE values.

============================================================
SUMMARY
============================================================
  Hampel outliers detected : 71
  Interpolation MAE (W):
    linear       13.6458
    ...

[check_preprocessing] All checks passed successfully.
```

---

## Étape 2 — Détection d'événements

### `Event_Detection/Algorithms/steady_states.py`

**Rôle** : Algorithme de Hart 1985 — détection des états stables et des transitions.

```bash
python Event_Detection/Algorithms/steady_states.py
```

**Résultat attendu** :
```
[find_steady_states] Processing 5000 samples …
[find_steady_states] Found 265 steady states, 3 significant transitions.
   power_active
0    199.85
1    598.73
...
```
Un tableau des états stables détectés et des transitions significatives (ΔPuissance > noise_level).

---

### `Event_Detection/Algorithms/cumsum.py`

**Rôle** : Détection de points de changement par CUSUM.

```bash
python Event_Detection/Algorithms/cumsum.py
```

**Résultat attendu** :
```
[CUSUM] Detected 12 change-points.
[ProbCUSUM] Detected 45 change-points.
```
- **Figure** : Signal avec les points de changement positifs (vert) et négatifs (rouge) annotés.

---

### `Event_Detection/Algorithms/ca_cfar.py`

**Rôle** : Détecteur CA-CFAR adaptatif.

```bash
python Event_Detection/Algorithms/ca_cfar.py
```

**Résultat attendu** :
```
[CA-CFAR] Detected 87 event samples (1.74% of signal).
```
- **Figure** : Signal avec les échantillons détectés en rouge.

---

### `Event_Detection/Algorithms/local_threshold_based.py`

**Rôle** : Seuillage local adaptatif avec détection de pics.

```bash
python Event_Detection/Algorithms/local_threshold_based.py
```

**Résultat attendu** :
```
[local_threshold] Detected 6 events.
  First 10 event indices: [302 345 801 856 2502 2547]
```
- **Figure** : 2 sous-graphiques — signal + marqueurs d'événements, et le signal `|ΔPower|`.

---

### `Event_Detection/Algorithms/binary_segmentation.py`

**Rôle** : Segmentation binaire + K-Means pour cycles ON/OFF.

```bash
python Event_Detection/Algorithms/binary_segmentation.py
```

**Résultat attendu** :
```
[binary_segmentation] 8 breakpoints, 3 complete cycles detected.
  Appliance  ON_Time  OFF_Time  Duration  Power_Level
 Dishwasher      200     1200      1000        700.15
 Dishwasher     2000     2800       800        499.87
 Dishwasher     3500     4500      1000        649.92
```
- **Figure 1** : Signal avec les points de rupture (orange).
- **Figure 2** : Scatter plot des clusters ON (rouge) vs OFF (bleu).
- **Figure 3** : Signal avec les cycles pairés surlignés en vert.

> **Note** : Ce script nécessite le package `ruptures`. Installez-le avec `pip install ruptures`.

---

## Étape 3 — Clustering

### `Clustering/Coding/cluster.py`

**Rôle** : K-Means avec sélection automatique du nombre de clusters.

```bash
python Clustering/Coding/cluster.py
```

**Résultat attendu** :
```
[cluster] Centroids (incl. OFF=0): [  0 150 300]
```
Les centroïdes représentent les niveaux de puissance typiques de chaque état de l'appareil.

---

### `Clustering/Coding/Appliance.py`

**Rôle** : Clustering en ligne et génération des paramètres HMM.

```bash
python Clustering/Coding/Appliance.py
```

**Résultat attendu** :
```
[refrigerator] Calculating cluster means …
pi['refrigerator']   = np.array([0.700000, 0.150000, 0.150000])
a['refrigerator']    = np.array([[0.950000, 0.030000, 0.020000], ...])
mean['refrigerator'] = np.array([[2.1500], [80.3200], [152.4700]])
cov['refrigerator']  = np.array([[[9.2100]], [[125.4300]], [[98.7600]]])
```
- **Figure 1** : Série de puissance avec les centroïdes en lignes pointillées.
- **Figure 2** : Index du cluster attribué à chaque instant.

Ces paramètres sont directement utilisables pour initialiser un SSHMM.

---

## Étape 4 — Désagrégation et Évaluation

### `HMM_Disaggregation/Super_State_HMM/Testing/test_Algorithm.py`

**Rôle** : Script principal d'évaluation de la désagrégation avec validation croisée.

```bash
python HMM_Disaggregation/Super_State_HMM/Testing/test_Algorithm.py \
    <test_id> <modeldb> <dataset> <precision> <measure> <denoised> <limit> <algo_name>
```

### Arguments

| Argument | Type | Exemple | Description |
|---|---|---|---|
| `test_id` | str | `exp01` | Identifiant de l'expérience |
| `modeldb` | str | `model_house1` | Nom du fichier modèle JSON (sans `.json`) |
| `dataset` | str | `House_1` | Nom du fichier CSV (sans `.csv`) |
| `precision` | int | `10` | Facteur de quantisation (divise les W) |
| `measure` | str | `W` | Unité : `W` (Watts) ou `A` (Ampères) |
| `denoised` | str | `denoised` | `denoised` ou `noisy` |
| `limit` | str | `5000` ou `all` | Nombre max d'observations (ou `all`) |
| `algo_name` | str | `SparseViterbi` | `Viterbi` ou `SparseViterbi` |

### Exemples d'utilisation

#### Exemple 1 — SparseViterbi (recommandé pour grands modèles)
```bash
python HMM_Disaggregation/Super_State_HMM/Testing/test_Algorithm.py \
    exp01 model_demo House_1 10 W denoised 5000 SparseViterbi
```

**Résultat attendu** :
```
[test_Algorithm] Starting  2026-03-26T21:05:14.229191
  dataset=House_1  modeldb=model_demo  precision=10  measure=W  denoised=denoised  limit=5000
[test_Algorithm] Loaded algorithm: algo_SparseViterbi

============================================================
Test ID   : exp01
Algorithm : SparseViterbi
Precision : 1/10 W
Folds     : 3
============================================================

Fold 1/3  |  Accuracy: 0.9234  |  MAE: {'ApplianceA': 12.3, 'ApplianceB': 8.7}  |  Time: 0.045s
Fold 2/3  |  Accuracy: 0.9187  |  MAE: {'ApplianceA': 14.1, 'ApplianceB': 9.2}  |  Time: 0.043s
Fold 3/3  |  Accuracy: 0.9301  |  MAE: {'ApplianceA': 11.8, 'ApplianceB': 8.4}  |  Time: 0.044s

============================================================
OVERALL ACCURACY : 0.9241
  MAE [ApplianceA] : 12.73 W
  MAE [ApplianceB] : 8.77 W
============================================================

CSV SUMMARY
test_id,algo,precision,folds,accuracy,mae,timestamp
exp01,SparseViterbi,10,3,0.9241,"ApplianceA=12.73; ApplianceB=8.77",2026-03-26 21:05:14

[plot] Saved: .../Testing/results/exp01_ApplianceA.png
[plot] Saved: .../Testing/results/exp01_ApplianceB.png
```

#### Exemple 2 — Viterbi classique
```bash
python HMM_Disaggregation/Super_State_HMM/Testing/test_Algorithm.py \
    exp02 model_demo House_1 10 W denoised 5000 Viterbi
```

Résultat identique au SparseViterbi mais potentiellement plus lent pour les modèles avec K > 100 super-états.

#### Exemple 3 — Quantisation plus fine (précision 5)
```bash
python HMM_Disaggregation/Super_State_HMM/Testing/test_Algorithm.py \
    exp03 model_demo House_1 5 W denoised all SparseViterbi
```

Avec `precision=5`, chaque incrément de quantisation correspond à 5 W (plus précis mais alphabet plus grand).

---

## Résultats générés automatiquement

Après chaque exécution de `test_Algorithm.py`, des graphiques PNG sont sauvegardés dans :
```
HMM_Disaggregation/Super_State_HMM/Testing/results/
├── exp01_ApplianceA.png   ← Consommation réelle vs estimée (Appareil A)
└── exp01_ApplianceB.png   ← Consommation réelle vs estimée (Appareil B)
```

Chaque graphique montre :
- **Bleu** : consommation réelle (vérité terrain)
- **Rouge** : consommation estimée par le SSHMM

---

## Métriques d'évaluation

| Métrique | Description | Valeur typique |
|---|---|---|
| **Accuracy** | % de super-états correctement identifiés | 85–95 % |
| **MAE (W)** | Erreur absolue moyenne de puissance par appareil | 5–30 W |
| **Temps/fold** | Temps d'inférence pour ~1666 observations | < 0.1s |

---

## Résolution des problèmes courants

### ModuleNotFoundError : numpy / pandas / etc.
```bash
pip install numpy pandas matplotlib scikit-learn scipy ruptures
```

### `ruptures` non installé (binary_segmentation.py)
```bash
pip install ruptures
```
Sans ce package, `binary_segmentation.py` affiche un avertissement et saute la démonstration.

### Aucune donnée REFIT disponible
Tous les scripts fonctionnent en mode démonstration avec des données synthétiques. Aucune action requise.

### Graphiques qui ne s'affichent pas (environnement headless/serveur)
Les scripts de visualisation utilisent `matplotlib.show()`. Dans un environnement sans affichage graphique (ex. serveur SSH), exécutez :
```bash
export MPLBACKEND=Agg
python <script>.py
```
Les graphiques seront alors sauvegardés en PNG au lieu d'être affichés.

---

## Exécution complète du pipeline (séquence recommandée)

```bash
# 0. Installer les dépendances
pip install numpy pandas matplotlib scikit-learn scipy ruptures

# 1. Valider le prétraitement
python Preprocessing/Algorithms/check_preprocessing.py

# 2. Tester la détection d'événements sur le signal agrégé
python Event_Detection/Algorithms/steady_states.py
python Event_Detection/Algorithms/cumsum.py

# 3. Tester le clustering sur un appareil
python Clustering/Coding/Appliance.py

# 4. Lancer l'évaluation complète de la désagrégation
python HMM_Disaggregation/Super_State_HMM/Testing/test_Algorithm.py \
    exp01 model_demo House_1 10 W denoised all SparseViterbi
```
