# PDR – Décision automatique pour la gestion de portefeuille financier

## Présentation du projet

Ce projet a été réalisé dans le cadre du PDR (*Prise de Décision Automatique*).  
Il porte sur l’étude et l’implémentation de stratégies de gestion dynamique de portefeuille dans un environnement financier incertain.

L’idée générale est de modéliser l’évolution du marché à l’aide d’outils probabilistes, puis de comparer plusieurs stratégies d’allocation de portefeuille, allant d’approches simples à une stratégie de décision automatique basée sur le **Monte Carlo Tree Search (MCTS)**.

Le projet a été développé de manière progressive, avec trois versions principales correspondant aux différentes étapes de notre travail :
- une première version entièrement simulée ;
- une deuxième version intégrant des données réelles ;
- une version finale optimisée, utilisée pour le benchmark technique final.

---

## Problématique

Le problème étudié est le suivant :

**Comment utiliser des modèles markoviens et des méthodes de décision automatique pour gérer dynamiquement un portefeuille financier dans un contexte incertain ?**

Nous cherchons plus précisément à comparer plusieurs stratégies d’allocation sur des données de marché, en tenant compte à la fois du rendement, du risque, et de la capacité d’adaptation de la stratégie.

---

## Objectifs

Les principaux objectifs du projet sont les suivants :

- modéliser un marché financier sous forme d’états de marché ;
- représenter la décision d’investissement comme un problème séquentiel ;
- implémenter plusieurs stratégies d’allocation ;
- comparer leurs performances sur des données réelles ;
- étudier l’apport d’une stratégie basée sur le MCTS ;
- proposer un benchmark technique propre et reproductible.

---

## Méthodes et concepts mobilisés

Ce projet s’appuie sur plusieurs notions vues en cours :

- **Processus de Décision Markoviens (MDP)**  
- **Chaînes de Markov** pour modéliser les régimes de marché  
- **Séries temporelles** pour exploiter les rendements financiers  
- **Monte Carlo Tree Search (MCTS)** pour la prise de décision  
- **Simulation** et **benchmark** de stratégies  
- **Parallélisation** des expériences via `multiprocessing`

Le marché est modélisé à partir de trois régimes :
- **Bear** : marché baissier
- **Flat** : marché neutre
- **Bull** : marché haussier

Ces états sont recalibrés à partir des rendements observés sur des fenêtres glissantes de données réelles.

---

## Données utilisées

Les données utilisées dans les versions avancées du projet proviennent de **Yahoo Finance** via la bibliothèque `yfinance`.

- Indice utilisé : **CAC 40** (`^FCHI`)
- Période étudiée dans la version finale : **2000–2026**
- Les simulations sont réalisées sur des **fenêtres aléatoires de 2000 jours**
- Une partie de la dynamique de marché est recalibrée localement à partir des rendements observés

Cela permet de passer d’un prototype purement simulé à un modèle plus proche de conditions réelles.

---

## Stratégies comparées

Trois stratégies principales sont comparées dans la version finale :

### 1. Marché passif (100 % actions)
Cette stratégie consiste à investir en permanence 100 % du portefeuille dans l’actif risqué.  
Elle sert de référence simple.

### 2. Sharpe Trigger (ST)
Cette stratégie ajuste l’allocation à partir d’un indicateur inspiré du ratio de Sharpe, calculé sur une fenêtre courte de rendements récents.

### 3. IA MCTS
Cette stratégie utilise une recherche arborescente de type **Monte Carlo Tree Search** pour choisir l’allocation à chaque étape.  
Elle constitue la partie la plus avancée du projet.

---

## Évolution du projet

Le dépôt contient trois versions du code, correspondant à l’évolution du travail.

### `src/PDR_v1.py`
Première version du projet.  
Cette version repose sur un marché entièrement simulé et sur des paramètres fixés manuellement.  
Elle permet de poser les bases du modèle et de comparer les premières stratégies.

### `src/PDR_v2_real_data.py`
Version intermédiaire.  
Cette version intègre des **données réelles** de marché via `yfinance` et calibre les rendements, volatilités et transitions à partir de l’historique observé.

### `src/final_model.py`
Version finale du projet.  
Elle introduit plusieurs améliorations techniques importantes :
- benchmark parallèle via `multiprocessing`
- recalibration dynamique de la chaîne de Markov
- espace d’actions discret pour améliorer la convergence
- comparaison finale MCTS vs Sharpe vs marché

**C’est cette version qui constitue la base du rendu technique final.**

---

## Structure du dépôt

```text
src/
├── PDR_v1.py
├── PDR_v2_real_data.py
└── final_model.py
