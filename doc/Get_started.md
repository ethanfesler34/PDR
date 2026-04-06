#  Get Started

## 1. Prérequis

Assurez-vous d’avoir installé :

 Python 3.9+
 pip

## 2. Installation des dépendances

Dans un terminal, installer les bibliothèques nécessaires :

pip install numpy matplotlib pandas yfinance tqdm

## 3. Lancer le programme

Depuis la racine du projet :
python src/final_model.py

## 4. Parametres importants

Le comportement du modèle dépend principalement de :

simu : nombre de simulations globales
nombre d’itérations Monte Carlo (MCTS)
Observations :
20 simulations
→ rapide mais peu fiable
200 simulations
→ bon compromis (recommandé)
2000 simulations
→ résultats robustes mais très long


## 5. Performance et matching

Le code utilise le multiprocessing :

plus de cœurs → plus rapide
ex :
20 cœurs → exécution fluide
10 cœurs → plus lent

 Adapter :

simu (ligne 207)
num_workers (ligne 209)

## 7. Recommandation

usage classique : 200 simulations

usage avancé : 1000+ simulations avec machine performante

ligne 164 concernée : prop = env_instance.mc_search(x, 0.4) où x est le nombre de simulations à régler
