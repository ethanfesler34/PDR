#!/usr/bin/env python3
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
import yfinance as yf
import pandas as pd

# --- STRUCTURES DE DONNÉES ---

class MCNode:
    def __init__(self, state, belief_particule, parent=None, action_taken=None):
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.n_visits = 0
        self.total_value = 0.0
        
    def get_mean(self):
        if self.n_visits == 0: return 0
        return self.total_value / self.n_visits
    
    def UCT_score(self, exploration_cst=1.0): # C=1.0 pour plus de focus avec 500 itérations
        if self.n_visits == 0: return float('inf')
        exploitation = self.get_mean()
        exploration = exploration_cst * np.sqrt(np.log(self.parent.n_visits) / self.n_visits)
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
        self.market_state_space = np.array([0, 1, 2]) # 0: Bear, 1: Flat, 2: Bull
        self.market_state = None
        self.market_price = None
        
        # Données sources (calculées dynamiquement par get_real_data)
        self.prices_data = None 
        self.ren_market = np.zeros(3) # Moyennes par état
        self.vol_market = np.zeros(3) # Volatilités par état
        self.market_transition = np.zeros((3, 3)) # Matrice de transition réelle
        
        self.rend_passive = 0.0001
        self.lambda_risk = 0.0005 
        self.t = 0 

    def reset(self):
        self.t = 0
        self.wealth = self.start_wealth
        self.market_state = random.choice([0, 1, 2])
        return (self.wealth, self.market_state)

    def action(self, prop, r_return):
        perf = prop * r_return + (1 - prop) * self.rend_passive
        old = self.wealth
        new = old * (1 + perf)
        self.wealth = new
        
        # Pénalité basée sur la vraie vol de l'état actuel
        vol_courante = self.vol_market[self.market_state]
        pen = self.lambda_risk * (prop**2) * (vol_courante**2)
        
        reward = (new - old) - pen
        return (self.wealth, self.market_state), reward

    def update_belief(self, current_belief, action, reward):
        trans = np.dot(current_belief, self.market_transition)
        vraisemblance = []
        for i in range(3):
            res = norm.pdf(reward, self.ren_market[i], self.vol_market[i])
            vraisemblance.append(res)
        belief = np.array(vraisemblance) * trans
        if np.sum(belief) > 1e-9:
            return belief / np.sum(belief)
        return trans

    def get_real_data(self, ticker_str, n_days=1000):
        data = yf.download(ticker_str, period="max", auto_adjust=True)
        data = data.tail(n_days + 1)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data['Returns'] = data['Close'].pct_change()
        returns = data['Returns'].dropna()
        self.prices_data = data['Close']
        
        vol_globale = returns.std()
        etats = []
        for r in returns:
            if r > 0.25 * vol_globale: etats.append(2)
            elif r < -0.25 * vol_globale: etats.append(0)
            else: etats.append(1)
        
        P = np.zeros((3, 3))
        for i in range(len(etats)-1):
            P[etats[i], etats[i+1]] += 1
        for i in range(3):
            if P[i].sum() > 0: P[i] = P[i] / P[i].sum()
            else: P[i] = np.array([1/3, 1/3, 1/3])
        self.market_transition = P

        etats_arr = np.array(etats)
        returns_arr = np.array(returns)
        for i in range(3):
            mask = (etats_arr == i)
            if np.any(mask):
                self.ren_market[i] = returns_arr[mask].mean()
                self.vol_market[i] = returns_arr[mask].std()
            else:
                self.ren_market[i], self.vol_market[i] = 0, vol_globale

        print("\n--- Calibration terminée sur données réelles ---")
        print(f"Matrice de Transition :\n{self.market_transition}")
        return etats, returns

    def mc_search(self, current_belief, iteration):
        root = MCNode(None, belief_particule=current_belief)
        for _ in range(iteration):
            start_state = np.random.choice(self.market_state_space, p=current_belief)
            self.simulate(start_state, root, 0)
        return max(root.children, key=lambda c: c.n_visits).action_taken

    def simulate(self, s, node, depth):
        if depth > 10: return 0 # Profondeur 10 pour limiter le bruit financier
        if len(node.children) < (node.n_visits**0.5) + 1:
            action = random.uniform(0, 1)
            child = MCNode(None, None, parent=node, action_taken=action)
            node.children.append(child)
            total_reward = self.rollout(s, 10 - depth)
        else: 
            best_child = max(node.children, key=lambda c: c.UCT_score())
            action = best_child.action_taken
            rend_fictif = np.random.normal(self.ren_market[s], self.vol_market[s])
            reward = (self.wealth * (1 + (action*rend_fictif + (1-action)*self.rend_passive)) - self.wealth) - (self.lambda_risk * action**2 * self.vol_market[s]**2)
            next_s = np.random.choice(self.market_state_space, p=self.market_transition[s])
            total_reward = reward + self.simulate(next_s, best_child, depth + 1)
        node.update(total_reward)
        return total_reward

    def rollout(self, s, depth):
        total = 0
        for _ in range(depth):
            action = random.uniform(0, 1)
            rend = np.random.normal(self.ren_market[s], self.vol_market[s])
            reward = (self.wealth * (1 + (action*rend + (1-action)*self.rend_passive)) - self.wealth) - (self.lambda_risk * action**2 * self.vol_market[s]**2)
            s = np.random.choice(self.market_state_space, p=self.market_transition[s])
            total += reward
        return total

# --- TEST ---

def tester_strategie(scenario, mode):
    etats, rends = scenario
    env.reset()
    env.market_price = env.prices_data.iloc[0]
    
    wealth_list = [env.wealth]
    market_history_plot = [env.market_price]
    current_belief = np.array([1/3, 1/3, 1/3])
    
    for i in range(len(rends)):
        if mode == "MCTS":
            prop = env.mc_search(current_belief, 500) # 500 itérations
        elif mode == "bas":
            prop = 1.0
        elif mode == "mid":
            prec_state = etats[i-1] if i > 0 else random.choice([0, 1, 2])
            prop = [0, 0.5, 1.0][prec_state]
        else: # Sharpe
            prop = strat_ST(rends.iloc[:i], 5)
            
        current_rend = rends.iloc[i]
        state, reward = env.action(prop, current_rend) 
        
        env.market_price = env.prices_data.iloc[i+1]
        env.market_state = etats[i]
        
        if mode == "MCTS":
            current_belief = env.update_belief(current_belief, prop, reward)
        
        wealth_list.append(env.wealth)
        market_history_plot.append(env.market_price)
        
    return wealth_list, market_history_plot

def strat_ST(rend, window):
    if len(rend) < window: return 1.0
    recent = rend[-window:]
    sharpe = (np.mean(recent) - 0.0001) / (np.std(recent) + 1e-6)
    return 1 / (1 + np.exp(-np.clip(sharpe, -20, 20)))

# --- EXECUTION ---

env = mdp()
S = env.get_real_data("^FCHI", n_days=1000)

w_bas, _ = tester_strategie(S, mode="bas")
w_mid, _ = tester_strategie(S, mode="mid")
w_st, _ = tester_strategie(S, mode="ST")
w_mcts, m_ref = tester_strategie(S, mode="MCTS")

# Affichage en deux graphiques séparés
fig, (ax_market, ax_strat) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax_market.plot(m_ref, color='red', linewidth=1.5, label="Indice CAC 40")
ax_market.set_title("Évolution du Marché (Environnement)", fontsize=14)
ax_market.set_ylabel("Points de l'Indice")
ax_market.grid(True, alpha=0.3)
ax_market.legend()

ax_strat.plot(w_bas, label="Stratégie: 100% Actions", color='blue', alpha=0.4)
ax_strat.plot(w_mid, label="Stratégie: Adaptative Mid", color='green', alpha=0.6)
ax_strat.plot(w_st, label="Stratégie: Sharpe Ratio", color='black', alpha=0.6)
ax_strat.plot(w_mcts, label="Stratégie: IA (MCTS)", color='orange', linewidth=2.5)
ax_strat.set_title("Performance des Stratégies (Fortune)", fontsize=14)
ax_strat.set_ylabel("Fortune ($)")
ax_strat.legend()
ax_strat.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Benchmark sur X simulations pour mettre en avant la meilleur strat
bas_l = []
mid_l = []
st_l = []
mcts_l = []
simu = 0

print(f"Lancement du Benchmark ({simu} simulations)...")

for i in range(0, simu):
    pourcentage = (i + 1) / simu * 100
    sys.stdout.write(f"\rProgression : [{int(pourcentage)}%] {'#' * int(pourcentage // 5)}{'.' * (20 - int(pourcentage // 5))}")
    sys.stdout.flush()

    S = env.get_real_data("^FCHI", n_days=1000)
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


