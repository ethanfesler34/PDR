#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:06:36 2025
@author: Ethan Matthieu Edmond Maxence
Benchmark MCTS Parallélisé
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import os  # Ajouté pour une détection plus robuste des coeurs

# Structure

class MCNode:
    def __init__(self, parent=None, action_taken=None):
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.n_visits = 0
        self.total_value = 0.0
        
    def get_mean(self):
        if self.n_visits == 0: return 0
        return self.total_value / self.n_visits
    
    def UCT_score(self, c_val):
        if self.n_visits == 0: return float('inf')
        return self.get_mean() + c_val * np.sqrt(np.log(self.parent.n_visits) / self.n_visits)

    def update(self, reward):
        self.n_visits += 1
        self.total_value += reward
        if self.parent:
            self.parent.update(reward)

# Environnement

class mdp:
    def __init__(self):
        self.start_wealth = 1000
        self.wealth = None
        self.market_state_space = np.array([0, 1, 2])
        self.market_state = None
        self.ren_market = np.zeros(3)
        self.vol_market = np.zeros(3)
        self.market_transition = np.zeros((3, 3))
        self.possible_actions = np.linspace(0, 1, 21) 
        self.rend_passive = 0.0001
        self.lambda_risk = 0.01 
        self.prev_prop = 0.0

    def reset(self):
        self.wealth = self.start_wealth
        self.market_state = 1
        self.prev_prop = 0.0
        return (self.wealth, self.market_state)

    def action(self, prop, r_return):
        perf = prop * r_return + (1 - prop) * self.rend_passive
        old = self.wealth
        moov = abs(prop - self.prev_prop)
        frais = old * moov * (0.00115 + (0.003 if prop > self.prev_prop else 0))
        new = old * (1 + perf) #- frais
        self.wealth = new
        self.prev_prop = prop
        vol_courante = self.vol_market[self.market_state]
        pen = self.lambda_risk * (prop**2) * (vol_courante**2)
        return (self.wealth, self.market_state), (new - old) - pen

    def recalibrer_dynamique(self, returns_slice):
        vol_globale = returns_slice.std()
        etats = [2 if r > 0.25*vol_globale else 0 if r < -0.25*vol_globale else 1 for r in returns_slice]
        P = np.zeros((3, 3))
        for i in range(len(etats)-1): P[etats[i], etats[i+1]] += 1
        for i in range(3):
            if P[i].sum() > 0: P[i] /= P[i].sum()
            else: P[i] = np.array([1/3, 1/3, 1/3])
        self.market_transition = P
        etats_arr = np.array(etats)
        for i in range(3):
            mask = (etats_arr == i)
            if np.any(mask):
                self.ren_market[i], self.vol_market[i] = returns_slice.iloc[mask].mean(), returns_slice.iloc[mask].std()
            else: self.ren_market[i], self.vol_market[i] = 0, vol_globale
        self.market_state = etats[-1]

    def mc_search(self, iteration, c_val):
        root = MCNode()
        for _ in range(iteration):
            start_state = np.random.choice(self.market_state_space, p=[1/3, 1/3, 1/3])
            self.simulate(start_state, root, 0, c_val, self.wealth) 
        return max(root.children, key=lambda c: c.n_visits).action_taken

    def simulate(self, s, node, depth, c_val, current_w):
        if depth > 12 or current_w <= 0: return 0
        if len(node.children) < (node.n_visits**0.5) + 1:
            action = random.choice(self.possible_actions)
            child = MCNode(node, action)
            node.children.append(child)
            prev_a = node.action_taken if node.action_taken is not None else self.prev_prop
            moov = action - prev_a
            f = current_w * abs(moov) * (0.00115 + (0.003 if moov > 0 else 0))
            rend = np.random.normal(self.ren_market[s], self.vol_market[s])
            new_w = current_w * (1 + (action * rend + (1-action)*self.rend_passive))
            reward = (new_w - current_w) - (self.lambda_risk * action**2 * self.vol_market[s]**2)
            next_s = np.random.choice(self.market_state_space, p=self.market_transition[s])
            total_reward = reward + self.rollout(next_s, 35 - depth, new_w)

        else: 
            best_child = max(node.children, key=lambda c: c.UCT_score(c_val))
            action = best_child.action_taken
            moov = action - (node.action_taken if node.action_taken else self.prev_prop)
            f = current_w * abs(moov) * (0.00115 + (0.003 if moov > 0 else 0))
            rend = np.random.normal(self.ren_market[s], self.vol_market[s])
            new_w = current_w * (1 + (action * rend + (1-action)*self.rend_passive))
            reward = (new_w - current_w) - (self.lambda_risk * action**2 * self.vol_market[s]**2)
            next_s = np.random.choice(self.market_state_space, p=self.market_transition[s])
            total_reward = reward + self.simulate(next_s, best_child, depth + 1, c_val, new_w)
            
        node.update(total_reward)
        return total_reward

    def rollout(self, s, depth, current_w): 
        if depth <= 0 or current_w <= 0: return 0
        t_w, c_a, total = current_w, self.prev_prop, 0
        for _ in range(depth):
            a = random.choice(self.possible_actions)
            f = t_w * abs(a-c_a) * (0.00115 + (0.003 if a > c_a else 0))
            new_w = t_w * (1 + (a*np.random.normal(self.ren_market[s], self.vol_market[s]) + (1-a)*self.rend_passive)) #- f
            total += (new_w - t_w) - (self.lambda_risk * a**2 * self.vol_market[s]**2)
            s = np.random.choice(self.market_state_space, p=self.market_transition[s])
            t_w, c_a = new_w, a
        return total

# Run

def run_strategy(mode, all_returns, all_prices, window_size, env_instance):
    env_instance.reset()
    wealth_list = [env_instance.wealth]
    for t in range(window_size, len(all_returns) - 1):
        if (t - window_size) % 10 == 0:
            env_instance.recalibrer_dynamique(all_returns.iloc[t - window_size : t])
        
        if mode == "MCTS":
            prop = env_instance.mc_search(2000, 0.4) 
            if abs(prop - env_instance.prev_prop) < 0.10: prop = env_instance.prev_prop
        elif mode == "ST":
            recent = all_returns.iloc[t-5:t]
            sharpe = (recent.mean() - 0.0001) / (recent.std() + 1e-6)
            prop = 1 / (1 + np.exp(-np.clip(sharpe, -20, 20)))
        else: prop = 1.0 

        env_instance.action(prop, all_returns.iloc[t])
        wealth_list.append(env_instance.wealth)
    return wealth_list

def run_one_simulation(args):
    all_returns_full, all_prices_full = args
    local_env = mdp()
    
    idx_start = random.randint(0, len(all_returns_full) - 2001)
    slice_ret = all_returns_full.iloc[idx_start : idx_start + 2000]
    slice_pri = all_prices_full.iloc[idx_start : idx_start + 2000]

    local_env.reset()
    local_env.recalibrer_dynamique(slice_ret.iloc[:1000])
    w_bas = run_strategy("bas", slice_ret, slice_pri, 1000, local_env)
    
    local_env.reset()
    local_env.recalibrer_dynamique(slice_ret.iloc[:1000])
    w_st  = run_strategy("ST", slice_ret, slice_pri, 1000, local_env)
    
    local_env.reset()
    local_env.recalibrer_dynamique(slice_ret.iloc[:1000])
    w_mcts = run_strategy("MCTS", slice_ret, slice_pri, 1000, local_env)

    return (
        ((w_bas[-1] / w_bas[0]) - 1) * 100,
        ((w_st[-1] / w_st[0]) - 1) * 100,
        ((w_mcts[-1] / w_mcts[0]) - 1) * 100
    )

# Execution

if __name__ == "__main__":
    # Multiprocessing
    logical_cores = os.cpu_count() or 4
    num_workers = max(1, logical_cores - 1) 
    
    simu = 8
    print(f"Initialisation")
    print(f"Machine : {logical_cores} cœurs détectés.")
    print(f"Charge : {simu} simulations prévues.")

    print("\nTéléchargement des données")
    data_full = yf.download("^FCHI", start="2000-01-01", end="2026-01-01", auto_adjust=True)
    
    # Nettoyage des colonnes yfinance
    if isinstance(data_full.columns, pd.MultiIndex):
        data_full.columns = data_full.columns.get_level_values(0)

    all_returns_full = data_full['Close'].pct_change().dropna()
    all_prices_full = data_full['Close']

    tasks = [(all_returns_full, all_prices_full) for _ in range(simu)]
    
    # maxtasksperchild permet de vider la RAM entre les simulations
    with mp.Pool(processes=num_workers, maxtasksperchild=5) as pool:
        results = list(tqdm(pool.imap_unordered(run_one_simulation, tasks), total=simu))

    # Traitement et graphique
    bas_perf = [r[0] for r in results]
    st_perf  = [r[1] for r in results]
    mcts_perf = [r[2] for r in results]

    plt.figure(figsize=(12, 6))
    indices = np.arange(1, simu + 1)
    width = 0.25
    plt.bar(indices - width, bas_perf, width, label='Marché (100% Actions)', color='blue', alpha=0.3)
    plt.bar(indices, st_perf, width, label='ST (Sharpe)', color='black', alpha=0.5)
    plt.bar(indices + width, mcts_perf, width, label='IA MCTS', color='orange')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f"Performance : MCTS vs ST vs Marché (1000j Run)")
    plt.ylabel("Rendement Total (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    mcts_vs_st = sum(np.array(mcts_perf) > np.array(st_perf))
    print(f"\nBilan")
    print(f"MCTS bat Sharpe (ST) : {mcts_vs_st}/{simu} ({mcts_vs_st/simu*100:.1f}%)")
    print(f"Rendement moyen MCTS : {np.mean(mcts_perf):.2f}%")
    print(f"Rendement moyen ST   : {np.mean(st_perf):.2f}%")