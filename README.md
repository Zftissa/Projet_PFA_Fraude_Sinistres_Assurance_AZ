# Détection de fraude dans les sinistres d'assurance par Machine Learning

## 1. Contexte

Ce projet est un projet Data Science / Intelligence Artificielle appliqué au secteur de l'assurance. Il vise à aider une compagnie d'assurance à identifier automatiquement les dossiers de sinistres suspects à partir des données disponibles.

Dans une compagnie d'assurance, les gestionnaires reçoivent plusieurs demandes d'indemnisation. Certaines peuvent être normales, tandis que d'autres peuvent contenir des signes de fraude. L'objectif n'est pas de remplacer le gestionnaire, mais de lui fournir un outil d'aide à la décision pour prioriser les dossiers à contrôler.

## 2. Problématique

**Comment détecter automatiquement les dossiers de sinistres suspects afin de réduire le risque de fraude dans une compagnie d'assurance ?**

## 3. Solution proposée

La solution consiste à développer un modèle de Machine Learning qui :

- analyse les données des sinistres ;
- apprend à distinguer les dossiers frauduleux des dossiers non frauduleux ;
- calcule une probabilité de fraude pour chaque dossier ;
- classe les dossiers par niveau de risque : faible, moyen ou élevé ;
- affiche les résultats dans un dashboard Streamlit interactif.

## 4. Dataset utilisé

Le dataset utilisé est : `insurance_claims.csv`.

Il contient **1000 dossiers** et **40 colonnes**. La colonne cible est : `fraud_reported`.

- `Y` : dossier frauduleux ;
- `N` : dossier non frauduleux.

Répartition de la cible :

- Fraude (`Y`) : 247 dossiers ;
- Non fraude (`N`) : 753 dossiers.

La classe fraude est minoritaire, ce qui est normal dans un contexte réel de fraude. Pour cette raison, le **recall de la classe fraude** est très important.

## 5. Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Matplotlib

## 6. Architecture du projet

```text
Projet_PFA_Fraude_Sinistres_Assurance_AZ/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── insurance_claims.csv
│
├── outputs/
│   ├── fraud_detection_model.joblib
│   ├── predictions.csv
│   ├── metrics.json
│   └── feature_importance.csv
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── risk_scoring.py
│   └── utils.py
│
├── notebooks/
│   └── exploration_data.ipynb
│
```

## 7. Méthodologie

### 7.1 Analyse exploratoire

Les premières étapes consistent à :

- charger le dataset ;
- vérifier la taille des données ;
- analyser les types de colonnes ;
- vérifier les valeurs manquantes ;
- analyser la variable cible `fraud_reported` ;
- comprendre les variables liées aux sinistres, aux assurés, aux véhicules et aux montants.

### 7.2 Prétraitement

Le prétraitement comprend :

- remplacement des valeurs `?` par des valeurs manquantes ;
- transformation de la cible : `Y = 1`, `N = 0` ;
- suppression des colonnes identifiantes ou inutiles comme `policy_number`, `insured_zip`, `incident_location` et `_c39` ;
- création de variables à partir des dates ;
- imputation des valeurs manquantes ;
- encodage des variables catégorielles avec OneHotEncoder ;
- standardisation des variables numériques ;
- création d'un pipeline Scikit-learn.

### 7.3 Modélisation

Plusieurs modèles sont testés :

- Logistic Regression ;
- Decision Tree ;
- Random Forest ;
- ExtraTreesClassifier ;
- GradientBoostingClassifier.

Les modèles sont comparés selon :

- Accuracy ;
- Precision fraude ;
- Recall fraude ;
- F1-score fraude ;
- ROC-AUC ;
- Confusion matrix.

Le meilleur modèle est choisi selon le F1-score de la classe fraude, puis le recall fraude, puis le ROC-AUC.

## 8. Scoring de risque

Pour chaque dossier, le modèle calcule une probabilité de fraude appelée `fraud_probability`.

Les niveaux de risque sont définis ainsi :

- **Faible** : probabilité < 0.40 ;
- **Moyen** : 0.40 <= probabilité < 0.70 ;
- **Élevé** : probabilité >= 0.70.

Chaque niveau de risque reçoit une action recommandée :

- Faible : traitement normal ;
- Moyen : vérification complémentaire ;
- Élevé : contrôle approfondi par gestionnaire.

## 9. Dashboard Streamlit

Le dashboard a été finalisé pour être plus clair pendant la soutenance. Il contient maintenant :

- une page organisée par onglets : **Vue générale**, **Analyse fraude**, **Dossiers suspects** et **Performance modèle** ;
- des KPIs : nombre de dossiers, fraudes suspectées, taux estimé, probabilité moyenne, risque élevé ;
- un **seuil de décision ajustable** dans la barre latérale pour montrer l'effet du seuil sur les dossiers suspectés ;
- des filtres interactifs par niveau de risque, probabilité, type de sinistre et montant ;
- des graphiques : répartition par risque, distribution des probabilités, top dossiers suspects ;
- l'importance des variables ;
- un tableau interactif des dossiers triés du plus suspect au moins suspect ;
- un bouton pour télécharger les résultats en CSV.

Remarque importante : la cible réelle `fraud_reported` n'est pas affichée dans le tableau métier des dossiers suspects, car dans un cas réel le gestionnaire ne connaît pas encore la vraie réponse. Elle est uniquement utilisée dans la partie **Performance modèle** pour l'évaluation.

## 10. Installation et exécution sur Windows PowerShell

### Étape 1 : accéder au projet

```powershell
cd Projet_PFA_Fraude_Sinistres_Assurance_AZ
```

### Étape 2 : créer un environnement virtuel

```powershell
python -m venv venv
```

### Étape 3 : activer l'environnement

```powershell
venv\Scripts\activate
```

### Étape 4 : installer les bibliothèques

```powershell
python -m pip install -r requirements.txt
```

Si Windows utilise le lanceur `py`, tu peux aussi faire :

```powershell
py -m pip install -r requirements.txt
```

### Étape 5 : entraîner le modèle

```powershell
python train_model.py
```

ou :

```powershell
py train_model.py
```

### Étape 6 : lancer le dashboard

```powershell
python -m streamlit run app.py
```

ou :

```powershell
py -m streamlit run app.py
```

## 11. Résultats

Après l'exécution de `train_model.py`, les résultats sont sauvegardés dans le dossier `outputs/` :

- `fraud_detection_model.joblib` : modèle entraîné ;
- `predictions.csv` : dossiers avec probabilité de fraude et niveau de risque ;
- `metrics.json` : métriques des modèles ;
- `feature_importance.csv` : variables importantes.

Résultat obtenu sur la version actuelle du dataset :

- Meilleur modèle : **Logistic Regression** ;
- Accuracy : **82%** ;
- Recall fraude : **69,39%** ;
- F1-score fraude : **65,38%** ;
- ROC-AUC : **83,04%**.

Dans ce projet, le recall fraude est particulièrement important, car il mesure la capacité du modèle à détecter les dossiers frauduleux.

## 12. Limites du projet

Ce projet reste un outil d'aide à la décision. Ses principales limites sont :

- le dataset contient 1000 dossiers, ce qui reste limité pour un déploiement réel ;
- le modèle détecte un risque, mais ne prouve pas juridiquement une fraude ;
- les seuils de risque doivent être validés avec des experts métier ;
- une compagnie réelle devrait intégrer plus d'historique client et plus de données opérationnelles.

## 13. Conclusion

Ce projet montre comment le Machine Learning peut aider une compagnie d'assurance à détecter les dossiers de sinistres suspects. Le dashboard permet aux gestionnaires de visualiser rapidement les dossiers à risque et de prioriser les contrôles.

Le modèle ne remplace pas l'expert métier. Il sert comme outil d'aide à la décision.

## 14. Perspectives

Les améliorations possibles sont :

- utiliser plus de données réelles ;
- ajouter des variables historiques sur les clients ;
- optimiser les seuils de risque avec les experts métier ;
- intégrer le dashboard dans le système interne de l'assurance ;
- ajouter un suivi des dossiers contrôlés après décision du gestionnaire.

## Pipeline Machine Learning demandé par l'encadrant

Le projet suit les étapes suivantes :

1. **Exploration des données** : analyse du fichier `insurance_claims.csv`, identification de la variable cible `fraud_reported` et étude de la répartition fraude / non fraude.
2. **Transformation et nettoyage** : traitement des valeurs manquantes, gestion des valeurs inconnues et transformation des variables catégorielles en variables numériques.
3. **Analyse des variables influentes** : génération du fichier `outputs/feature_importance.csv` afin d'identifier les variables qui influencent le plus la prédiction.
4. **Lancement de l'entraînement** : séparation de la target `fraud_reported` dans `y` et utilisation des autres colonnes dans `X`, puis entraînement et comparaison de plusieurs modèles.

Un notebook explicatif est disponible ici : `notebooks/01_pipeline_machine_learning.ipynb`.
