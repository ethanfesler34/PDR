# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:06:36 2025

@author: Ethan
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

class mdp:
    def __init__(self):
        self.start_wealth = 1000
        self.wealth = None
    
        self.market_state_space = np.array([0,1,2])                 # 0 si marché en baisse, 1 si neutre, 2 si hausse
        self.market_state = None
    
        self.prop_action = np.array([0, 0.5, 1])                    # choix du pourcentage d'investissement en actions (0%, 50% ou 100%) --> à modifier pour que l'agent puisse choisir entre 0 et 1
    
        self.ren_market = np.array([-0.02, 0.01, 0.05])             # rendement du marché en fonction de l'état
    
        self.vol_market = np.array([0.10, 0.05, 0.08])              # volatitlité du marché en fonction de l'état
    
        market_transition = np.array([                              # matrice de Transitions entre état du marché
                [0.7, 0.2, 0.1],                                    # baisse --> (baisse, neutre, hausse)
                [0.2, 0.6, 0.2],                                    # neutre --> (baisse, neutre, hausse)
                [0.1, 0.3, 0.6]])                                   # hausse --> (baisse, neutre, hausse)
        self.market_transition = market_transition
        
        self.rend_passive = 0.03
        
        self.lambda_risk = 5e-3
    
        self.t = 0  

        self.perf = None                                               # date
            
        
    def reset(self):
        
        self.t = 0
        self.wealth = self.start_wealth
        
        m = random.choice([0, 1, 2])
        self.market_state = m
        
        etat = (self.wealth, self.market_state)
        
        return etat
    
    
    def action(self, prop):
    
        
        # Proportion en action
        actions = self.prop_action[prop]
        
        # Etat actuel du marché (rend + vol)
        m_state = self.market_state
        rend = self.ren_market[m_state]
        vol = self.vol_market[m_state]
        
        
        # Rendement de la proportion d'actions en fonction de l'état du marché
        rend_f = np.random.normal(rend, vol)
        
        # Performance du portefeuille
        perf = actions*rend_f + (1-actions)*self.rend_passive
        self.perf = perf
        
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
    


# Initialisation de l'envionnement de la classe 'mdp' + reset  
env = mdp()         


tot_reward = 0
market_view = []
time = []
wealth = []
new = 0
market_state_l = []                     # Pour garder le même marché entre les strats

# Strat basique de toujours mettre tous en actions pendant 'i' jours
state = env.reset() 
for i in range (0, 100):
    state, reward = env.action(2)
    market_view.append(new + env.perf)
    market_state_l.append(state[1])
    time.append(env.t)
    new = env.perf
    tot_reward += reward
    wealth.append(env.wealth)

print("--------------------------Strat basique--------------------------")
print(f"Final Wealth = {state[0]}€")

plt.figure()
plt.plot(time, wealth)
plt.fill_between(time,wealth,alpha=0.3,color='blue')
plt.title("Wealth Basique Strat")



# Strat --> 0% si baisse, 50% si neutre, 100% si hausse
state = env.reset() 
wealth.clear()
for e in market_state_l:
    state = e
    env.market_state = state
    if state == 0:
        state, reward = env.action(0)
        wealth.append(env.wealth)
    elif state == 1:
        state, reward = env.action(1)
        wealth.append(env.wealth)
    else:
        state, reward = env.action(2)
        wealth.append(env.wealth)
    
print("\n--------------------------Strat basique+--------------------------")
print(f"Final Wealth = {state[0]}€")

plt.figure()
plt.plot(time, wealth)
plt.fill_between(time,wealth,alpha=0.3,color='blue')
plt.title("Wealth Basique+ Strat")

    

plt.figure()
plt.plot(time, market_view)
plt.fill_between(time,market_view,alpha=0.3,color='blue')
plt.title("Market View")



        
        
        
        
        
