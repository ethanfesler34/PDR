# PDR – Décision automatique pour la gestion de portefeuille

## Auteurs
Ethan Fesler
Matthieu Ringard
Edmond Bessi
Maxence Devalland

IMT Nord Europe – Parcours RAISE – FISE 2027


## Description

Ce projet explore l’utilisation de méthodes de décision automatique pour la gestion de portefeuille financier dans un environnement incertain.

L’approche repose sur :
- des modèles markoviens (MDP),
- des données financières réelles (CAC 40),
- et une stratégie basée sur le Monte Carlo Tree Search (MCTS).

L’objectif est de comparer différentes stratégies d’investissement et d’évaluer l’apport d’une approche basée sur l’exploration.


## Accès rapide

Code principal :

src/final_model.py

Documentation :
motivation (état de l’art synthétique),
guide d’utilisation (Get Started) et 
rapport technique complet (PDF)


## Structure du projet

```text
.
├── src/        # Codes sources
│   ├── PDR_v1.py
│   ├── PDR_v2_real_data.py
│   └── final_model.py
│
├── doc/        # Documentation
│   ├── motivation.md
│   ├── get_started.md
│   └── rapport_technique.pdf


