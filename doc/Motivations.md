## Motivation - Etats de l'art

**Auteurs :**  
Ethan Fesler, Matthieu Ringard, Edmond Bessi, Maxence Devalland  


La gestion de portefeuille financier constitue un problème complexe, en raison de l’incertitude sur l’évolution des marchés et du caractère séquentiel des décisions. À chaque instant, un investisseur doit adapter son allocation d’actifs à partir d’informations partielles, tout en cherchant à optimiser la performance globale sur un horizon de long terme.

Ce type de problématique s’inscrit naturellement dans le cadre de la **décision séquentielle sous incertitude**, où les choix effectués à un instant donné influencent directement les résultats futurs. Les **processus de décision markoviens (MDP)** offrent un cadre théorique adapté pour formaliser ces situations, en modélisant les états du système, les actions possibles, les transitions probabilistes ainsi que les récompenses associées.

Dans un contexte financier, l’évolution du marché peut être représentée à l’aide de **chaînes de Markov**, en considérant différents régimes (marché haussier, neutre ou baissier), chacun caractérisé par des comportements distincts en termes de rendement et de volatilité. Toutefois, ces régimes ne sont généralement pas directement observables, ce qui conduit à un cadre plus réaliste de type **POMDP**, dans lequel l’agent raisonne à partir d’une distribution de croyance sur les états possibles.

Par ailleurs, ce travail s’inscrit directement dans la continuité des enseignements suivis cette année, notamment en **séries temporelles**, en **optimisation numérique** et en **chaînes de Markov**. L’évolution des rendements financiers est en effet modélisée comme un processus temporel, présentant des dépendances dans le temps et une forte variabilité, ce qui constitue un point central de l’analyse.

Enfin, des méthodes de décision automatique plus avancées, comme le **Monte Carlo Tree Search (MCTS)**, permettent d’explorer efficacement différentes stratégies en simulant de nombreuses trajectoires possibles du système. Ces approches offrent un compromis pertinent entre exploration et exploitation dans des environnements incertains.

Dans le cadre de ce projet, une partie du travail de recherche documentaire a été facilitée par l’utilisation d’outils d’intelligence artificielle, notamment pour identifier rapidement des concepts clés, des références académiques et des ressources pédagogiques pertinentes. Ces éléments ont ensuite été approfondis et croisés avec les supports de cours afin de construire une compréhension cohérente du sujet.

Par exemple, certaines notions générales ont été vérifiées et complétées à l’aide de ressources accessibles comme :

 Processus de décision markovien :  
  https://fr.wikipedia.org/wiki/Processus_de_d%C3%A9cision_markovien  

 Chaînes de Markov :  
  https://fr.wikipedia.org/wiki/Cha%C3%AEne_de_Markov  

 Séries temporelles :  
  https://fr.wikipedia.org/wiki/S%C3%A9rie_temporelle  

 Méthodes de Monte Carlo :  
  https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Monte-Carlo  

 Monte Carlo Tree Search :  
  https://en.wikipedia.org/wiki/Monte_Carlo_tree_search  

Ce projet propose ainsi une modélisation simplifiée de la gestion de portefeuille, combinant modèles markoviens, analyse de séries temporelles et méthodes de simulation Monte Carlo, afin de comparer différentes stratégies de décision dans un cadre contrôlé.


## Références bibliographiques

 Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.  
 Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.  
 Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.  
 Browne, C. B. et al. (2012). *A Survey of Monte Carlo Tree Search Methods*. IEEE Transactions on Computational Intelligence.  
 Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.  
