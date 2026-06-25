# Script de soutenance - Projet PFA

## 1. Introduction

Bonjour,

Je vais vous présenter mon projet intitulé : **Détection de fraude dans les sinistres d’assurance par Machine Learning**.

Ce projet s’inscrit dans le domaine de la Data Science appliquée au secteur de l’assurance. L’objectif principal est d’aider une compagnie d’assurance à identifier automatiquement les dossiers de sinistres suspects.

## 2. Présentation du sujet

Dans une compagnie d’assurance, les clients déclarent des sinistres pour demander une indemnisation. Ces sinistres peuvent concerner des accidents, des vols de véhicules, des collisions ou d’autres dommages.

Cependant, certains dossiers peuvent contenir des informations fausses ou exagérées. Cela représente un risque financier pour l’assurance.

## 3. Problématique

La problématique de ce projet est la suivante :

**Comment détecter automatiquement les dossiers de sinistres suspects afin de réduire le risque de fraude dans une compagnie d’assurance ?**

## 4. Solution proposée

Pour répondre à cette problématique, j’ai développé une solution basée sur le Machine Learning.

La solution consiste à analyser les données des sinistres, entraîner plusieurs modèles de classification, calculer une probabilité de fraude pour chaque dossier, puis afficher les résultats dans un dashboard interactif.

## 5. Dataset utilisé

Le dataset utilisé s’appelle `insurance_claims.csv`.

Il contient 1000 dossiers de sinistres et 40 colonnes. La variable cible est `fraud_reported`, avec deux valeurs :

- `Y` pour les dossiers frauduleux ;
- `N` pour les dossiers non frauduleux.

Le dataset contient 247 dossiers frauduleux et 753 dossiers non frauduleux.

## 6. Méthodologie

La méthodologie suivie se compose de plusieurs étapes.

D’abord, j’ai chargé et analysé les données afin de comprendre les colonnes, les types de variables, les valeurs manquantes et la répartition de la cible.

Ensuite, j’ai réalisé le prétraitement des données. J’ai remplacé les valeurs manquantes, transformé la cible en format binaire, supprimé certaines colonnes identifiantes et encodé les variables catégorielles.

Après cela, j’ai entraîné plusieurs modèles de Machine Learning, notamment Logistic Regression, Decision Tree, Random Forest, ExtraTreesClassifier et GradientBoostingClassifier.

## 7. Évaluation du modèle

Les modèles ont été comparés avec plusieurs métriques : accuracy, precision, recall, F1-score et ROC-AUC.

Dans ce projet, le recall de la classe fraude est très important, car l’objectif est de détecter le maximum de dossiers suspects.

Le meilleur modèle est sélectionné automatiquement selon le F1-score de la classe fraude, puis le recall fraude et le ROC-AUC.

## 8. Scoring de risque

Après la prédiction, chaque dossier reçoit une probabilité de fraude appelée `fraud_probability`.

Ensuite, les dossiers sont classés en trois niveaux de risque :

- risque faible ;
- risque moyen ;
- risque élevé.

Chaque niveau de risque est associé à une action recommandée : traitement normal, vérification complémentaire ou contrôle approfondi par gestionnaire.

## 9. Présentation du dashboard

Le dashboard a été développé avec Streamlit.

Il permet de visualiser :

- le nombre total de dossiers ;
- le nombre de fraudes suspectées ;
- le taux de fraude estimé ;
- la probabilité moyenne de fraude ;
- les dossiers à risque élevé ;
- la répartition des dossiers par niveau de risque ;
- les top dossiers les plus suspects ;
- l’importance des variables du modèle.

Le dashboard contient aussi des filtres pour faciliter l’analyse par le gestionnaire.

## 10. Conclusion

En conclusion, ce projet montre comment le Machine Learning peut aider une compagnie d’assurance à détecter les dossiers suspects et à prioriser les contrôles.

Le modèle ne remplace pas la décision humaine, mais il représente un outil d’aide à la décision pour améliorer l’efficacité du traitement des sinistres.

## 11. Perspectives

Comme perspectives, on peut améliorer le projet en utilisant plus de données réelles, en optimisant les seuils de risque avec des experts métier, et en intégrant le dashboard dans le système interne de la compagnie d’assurance.
