# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:06:36 2025

@author: Ethan Matthieu Edmond Maxence
"""

"""
PDR : maximiser le gain à long terme en gérant son portefeuille par décision automatique


Processus de décision Markovien : 

Etats S : richesse actuelle, état du marché 

Action A : choix du pourcentage de l'allocation en actions (risqué) et en passif (non-risqué) 

Transition : la richesse a date t+1 dépend de l'état du marché (perte si baisse, gain si hausse) et du taux de rendement du passif 
--> faire varier le taux de rendement en fonction de l'état du marché (loi normal) --> faire suivre l'évolution du marché par une chaine de Markov 
--> taux fixe pour le passif 

Recompense R : en fonction de la performance entre t et t+1 
--> possibilité de pénaliser en fonction du risque R = wt+1 - Wt - coef_pénalité*volatitlité portefeuille (se baser sur volatilité marché) 



Objectif : comparer plusieurs stratégies avec ajout de réseaux bayésien plus tard 
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

# --- STRUCTURES DE DONNÉES ---

class MCNode:
    def __init__(self, state, belief_particule, parent = None, action_taken = None):
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.n_visits = 0
        self.total_value = 0.0
        self.belief_particule = belief_particule
        
    def get_mean(self):
        if self.n_visits == 0:
            return 0
        return self.total_value/self.n_visits
    
    def UCT_score(self, exploration_cst = np.sqrt(2)):
        if self.n_visits == 0:
            return float('inf')
        exploitation = self.get_mean()
        exploration = exploration_cst + np.sqrt(np.log(self.parent.n_visits)/self.n_visits)
        return exploitation + exploration

    def update(self, reward):
        self.n_visits += 1
        self.total_value += reward
        if self.parent:
            self.parent.update(reward)

# --- ENVIRONNEMENT ET SOLVEUR ---

class mdp:
    def __init__(self):
        self.start_wealth = 1000
        self.wealth = None
    
        self.market_state_space = np.array([0, 1, 2])               # 0: Baisse (Bear), 1: Neutre (Flat), 2: Hausse (Bull)
        self.market_state = None
        
        self.market_price = 100                                     # Indice de référence (S&P 500 par ex.)
        self.market_history = []
    
        self.prop_action = np.array([0, 0.5, 1])                    # Choix discrets (utilisés pour les strats simples)
    
        self.ren_market = np.array([-0.0015, 0.0002, 0.0008]) 
    
        self.vol_market = np.array([0.025, 0.008, 0.012]) 
    
        self.market_transition = np.array([
                [0.90, 0.08, 0.02],                                 # Baisse -> (reste en baisse, neutre, hausse)
                [0.05, 0.90, 0.05],                                 # Neutre -> (baisse, neutre, hausse)
                [0.02, 0.08, 0.90]])                                # Hausse -> (baisse, neutre, hausse)
        
        self.rend_passive = 0.0001                                  # Rendement sans risque (env. 2.5%/an)
        
        self.lambda_risk = 0.0002 
    
        self.t = 0                                                  # Compteur de jour 
            
        
    def reset(self):
        self.t = 0
        self.wealth = self.start_wealth
        self.market_price = 100
        self.market_history = [self.market_price]
        m = random.choice([0, 1, 2])
        self.market_state = m
        etat = (self.wealth, self.market_state)
        return etat
    
    
    def action(self, prop):
        # Proportion en action
        actions = prop
        
        # Etat actuel du marché (rend + vol)
        m_state = self.market_state
        rend = self.ren_market[m_state]
        vol = self.vol_market[m_state]
        
        # Rendement de la proporation d'actions en fonction de l'état du marché
        rend_f = np.random.normal(rend, vol)
        
        # MAJ du prix de l'indice
        self.market_price *= (1 + rend_f)
        self.market_history.append(self.market_price)
        
        # Performance du portefeuille
        perf = actions*rend_f + (1-actions)*self.rend_passive
        
        # Actualisation de la richesse
        old = self.wealth
        new = old*(1+perf)
        self.wealth = new
        
        # Pénalité de risque
        pen = self.lambda_risk*(actions**2)*(vol**2)
        
        # Récompense
        gain = new - old
        reward = gain - pen
        
        # Changement d'état du marché
        prob = self.market_transition[m_state]
        next_market_state = np.random.choice(self.market_state_space, p = prob)
        self.market_state = next_market_state
        
        self.t += 1
        
        state = (self.wealth, self.market_state)
        
        return state, reward
    
    
    def update_belief(self, current_belief, action, reward):
        trans = np.dot(current_belief, self.market_transition)
        vraisemblance  = []
        for i in range (len(self.market_state_space)):
            mu = self.ren_market[i]
            sigma = self.vol_market[i]
            
            res = norm.pdf(reward, mu, sigma)
            vraisemblance.append(res)
            
        belief = np.array(vraisemblance) * trans
        
        if np.sum(belief) > 0:
            belief = belief/np.sum(belief)
        else:
            belief = trans
            
        return belief
    
    def mc_search(self, current_belief, iteration):
        root = MCNode(None, belief_particule=current_belief, parent=None, action_taken=None) 
        
        for i in range(iteration):
            start_state = np.random.choice(self.market_state_space, p = current_belief)
            self.simulate(start_state, root, 0)
            
        best_child = max(root.children, key = lambda c : c.n_visits)
        
        return best_child.action_taken
            
            
    def simulate(self, s, node, depth):
        if depth > 10:              # 10 -->5 si machine pas trop puissante
            return 0
        if len(node.children) < (node.n_visits**0.5) + 1:
            action = random.uniform(0, 1)
            child = MCNode(state=None, belief_particule=None, parent=node, action_taken=action)
            node.children.append(child)
            total_reward = self.rollout(s, 10 - depth)
        else: 
            best_child = max(node.children, key = lambda c: c.UCT_score())
            action = best_child.action_taken
            
            rend_fictif = np.random.normal(self.ren_market[s], self.vol_market[s])
            perf = action*rend_fictif + (1-action)*self.rend_passive
            
            old = self.wealth
            new = old*(1+perf)              # Pas de MAJ de la fortune réel car c'est une simulation pour justement savoir quoi faire
            
            pen = self.lambda_risk*(action**2)*(self.vol_market[s]**2)
            gain = new - old
            
            reward = gain - pen
            
            prob = self.market_transition[s]
            next_s = np.random.choice(self.market_state_space, p = prob)
            
            total_reward = reward + self.simulate(next_s, best_child, depth+1)              
        
        node.update(total_reward)
        
        return total_reward
        
    def rollout(self, s, depth):
        total_reward = 0
        for i in range(depth):
            action = random.uniform(0, 1)
            rend_fictif = np.random.normal(self.ren_market[s], self.vol_market[s])
            perf = action*rend_fictif + (1-action)*self.rend_passive
            
            old = self.wealth
            new = old*(1+perf)              # Pas de MAJ de la fortune réel car c'est une simulation pour justement savoir quoi faire
            
            pen = self.lambda_risk*(action**2)*(self.vol_market[s]**2)
            gain = new - old
            
            reward = gain - pen
            
            prob = self.market_transition[s]
            s = np.random.choice(self.market_state_space, p = prob)
            
            total_reward += reward
            
        return total_reward
            
            
            


# --- TESTS ET BENCHMARKS ---

# Initialisation de l'envionnement de la classe 'mdp' + reset  
env = mdp()         
def generer_scenario(j):
    env.reset()
    etats_marche = []
    rendements_marche = []
    
    current_m_state = env.market_state
    for _ in range(j):
        etats_marche.append(current_m_state)
        # On tire le rendement selon l'état actuel
        rend = np.random.normal(env.ren_market[current_m_state], env.vol_market[current_m_state])
        rendements_marche.append(rend)
        
        # On calcule l'état suivant pour la boucle d'après
        prob = env.market_transition[current_m_state]
        current_m_state = np.random.choice(env.market_state_space, p=prob)
        
    return etats_marche, rendements_marche

def tester_strategie(scenario, mode="bas"):
    etats, rends = scenario
    env.reset()
    wealth_list = [env.wealth]
    market_history = [100]
    current_belief = np.array([1/3, 1/3, 1/3])
    
    for i in range(len(rends)):
        
        if mode == "bas":
            prop = 1.0
        elif mode == "mid":
            if i > 0:
                prec_state = etats[i-1]
            else:
                prec_state = random.choice([0, 1, 2])
            prop = [0, 0.5, 1.0][prec_state]
        elif mode == "ST":
            rendements_observes = rends[:i]
            prop = strat_ST(rendements_observes, 5)
        elif mode == "MCTS":
            prop = env.mc_search(current_belief, 200)               # 200 --> 100 sur machine pas trop puissante
            
        
        state, reward = env.action(prop) 
        
        if mode == "MCTS":
            current_belief = env.update_belief(current_belief, prop, reward)
        
        wealth_list.append(env.wealth)
        market_history.append(env.market_price)
        
    return wealth_list, market_history

def strat_ST(rend, window):
    if len(rend) < window:  # Si pas assez d'infos sur les rendements précédents
            action_idx = 1
    else:
        recent = rend[-window:]
        moyenne_mobile = np.mean(recent)
        volatilite_mobile = np.std(recent)
        
        # Permet de chosiir exactement la proportion à investir (au lieu de 0%, 50% ou 100%)
        action_idx = (moyenne_mobile-env.rend_passive)/volatilite_mobile**2         # Rapport de Sharpe
        action_idx = 1 / (1 + np.exp(-action_idx))                                   # Fonction sigmoide --> normalisation entre 0 et 1
    return action_idx

# --- EXECUTION ET AFFICHAGE ---

# On fixe un scénario unique
S = generer_scenario(1000)

# On fait tourner les strats sur ce scénario
wealth_bas, market_ref = tester_strategie(S, mode="bas")
wealth_mid, _ = tester_strategie(S, mode="mid")
wealth_ST, _ = tester_strategie(S, mode="ST")
wealth_MCTS, _ = tester_strategie(S, mode = "MCTS")

# Plot unique
plt.figure(figsize=(12, 6))

# Axe gauche pour la fortune
ax1 = plt.gca()
ax1.plot(wealth_bas, label="Stratégie: 100% Actions", color='blue')
ax1.plot(wealth_mid, label="Stratégie: Adaptative (Mid)", color='green')
ax1.plot(wealth_ST, label="Stratégie: Rapport de Sharpe", color = 'k')
ax1.plot(wealth_MCTS, label="Stratégie: MCTS (POMCP)", color='orange', linewidth=2)
ax1.set_ylabel("Fortune ($)")
ax1.legend(loc='upper left')

# Axe droit pour le marché (pour bien voir la corrélation)
ax2 = ax1.twinx()
ax2.plot(market_ref, label="Marché (Indice)", color='red', linestyle='--', alpha=0.5)
ax2.set_ylabel("Prix du Marché")
ax2.legend(loc='upper right')

plt.title("Comparaison des stratégies sur le même scénario de marché")
plt.show()

# Benchmark sur X simulations pour mettre en avant la meilleur strat
bas_l = []
mid_l = []
st_l = []
mcts_l = []
simu = 20

print(f"Lancement du Benchmark ({simu} simulations)...")

for i in range(0, simu):
    pourcentage = (i + 1) / simu * 100
    sys.stdout.write(f"\rProgression : [{int(pourcentage)}%] {'#' * int(pourcentage // 5)}{'.' * (20 - int(pourcentage // 5))}")
    sys.stdout.flush()

    S = generer_scenario(1000)
    wealth_bas, _ = tester_strategie(S, mode="bas")
    wealth_mid, _ = tester_strategie(S, mode="mid")
    wealth_ST, _ = tester_strategie(S, mode="ST")
    wealth_MCTS, _ = tester_strategie(S, mode="MCTS")
    
    bas_l.append(wealth_bas[-1])
    mid_l.append(wealth_mid[-1])
    st_l.append(wealth_ST[-1])
    mcts_l.append(wealth_MCTS[-1])

print("\n\nCalcul terminé ! Génération des graphiques...")

# Calcul des bénéfices
sum_bas = np.sum(bas_l) - simu * env.start_wealth
sum_mid = np.sum(mid_l) - simu * env.start_wealth
sum_st = np.sum(st_l) - simu * env.start_wealth
sum_mcts = np.sum(mcts_l) - simu * env.start_wealth

# Affichage du Bar Chart final
plt.figure()
plt.bar(["100% Strat", "Adaptative Mid", "ST Strat", "MCTS Strat"], [sum_bas, sum_mid, sum_st, sum_mcts], color=['blue', 'green', 'black', 'orange'])
plt.title(f"Bénéfices totaux après {simu} simulations")
plt.ylabel("Somme des bénéfices ($)")
plt.show()