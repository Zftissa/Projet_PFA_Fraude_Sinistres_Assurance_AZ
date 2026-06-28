# Pipeline demandé par l'encadrant

Ce dossier contient le notebook `01_pipeline_machine_learning.ipynb` qui montre les étapes suivantes :

1. **Exploration de données**  
   Analyse du dataset `insurance_claims.csv`, des colonnes, de la target `fraud_reported`, des valeurs manquantes et de la répartition fraude / non fraude.

2. **Transformation et nettoyage**  
   Remplacement des valeurs inconnues, traitement des valeurs manquantes et transformation des variables catégorielles en variables numériques avec `OneHotEncoder`.

3. **Analyse des variables influentes**  
   Lecture du fichier `outputs/feature_importance.csv` pour identifier les variables qui influencent le plus la prédiction.

4. **Lancement de l'entraînement**  
   Séparation du dataset en :
   - `X` : variables explicatives sans `fraud_reported`
   - `y` : variable cible `fraud_reported`

   Ensuite, plusieurs modèles sont entraînés et comparés.