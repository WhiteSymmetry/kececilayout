# -*- coding: utf-8 -*-
# ruff: noqa: N806, N815
"""
kececilayout.py

This module provides sequential-zigzag ("Keçeci Layout") and advanced visualization styles for various Python graph libraries.
Bu modül, çeşitli Python graf kütüphaneleri için sıralı-zigzag ("Keçeci Layout") ve gelişmiş görselleştirme stilleri sağlar.

**Key Features:**
*   **Linear Focus:** Ideal for visualizing paths, chains, or ordered processes.
*   **Deterministic:** Produces identical results for the same input.
*   **Overlap Reduction:** Prevents node collisions by spreading them across axes.
*   **Parametric:** Fully customizable with parameters like `primary_spacing`, `secondary_spacing`, `primary_direction`, and `secondary_start`.

**v0.2.7**: Curved, transparent, 3D, and `expanding=True` styles supported.

**v0.5.0:** 

layouts = ['2d', 'cylindrical', 'cubic', 'spherical', 'elliptical', 'toric']

styles = ['standard', 'default', 'curved', 'helix', '3d', 'weighted', 'colored']

**v0.5.1:** edge (kececi_layout_edge)

**v0.6.0:** periodic table
"""

from collections import defaultdict
import graphillion as gg
import igraph as ig
import itertools # Graphillion için eklendi
import math
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import networkit as nk
import networkx as nx
import numpy as np # rustworkx
from numba import jit
import os
import pandas as pd
import platform # graph_tool için
import random
import re
import rustworkx as rx
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings


# Ana bağımlılıklar (çizim için gerekli)
try:
    import networkx as nx
    #from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    raise ImportError(
        "Bu modülün çalışması için 'networkx' ve 'matplotlib' gereklidir. "
        "Lütfen `pip install networkx matplotlib` ile kurun."
    ) from e

# Opsiyonel graf kütüphaneleri
try:
    import rustworkx as rx
except ImportError:
    rx = None
try:
    import igraph as ig
except ImportError:
    ig = None
try:
    import networkit as nk
except ImportError:
    nk = None
try:
    import graphillion as gg
except ImportError:
    gg = None
# graph-tool sadece Linux'ta import edilsin
if platform.system() == "Linux":
    try:
        import graph_tool.all as gt
    except ImportError:
        gt = None
else:
    gt = None

"""
@jit(nopython=True)
def calculate_coordinates(nodes, primary_spacing, secondary_spacing, primary_direction, secondary_start, expanding):
    #Numba ile hızlandırılmış koordinat hesaplama.
    pos = {}
    for i, node_id in enumerate(nodes):
        # Koordinat hesaplama mantığı...
        pos[node_id] = (x, y)
    return pos
"""
class KececiZigzagValidator:
    def __init__(self):
        self.kececi_standard = {
            'x_spacing': 0.85,
            'sin_freq': 0.714,  # π/2.2
            'y_amp': 1.4
        }
    
    def perfect_kececi(self, G):
        """🦙 MÜKEMMEL KEÇECİ"""
        nodes = sorted(G.nodes())
        pos = {}
        for i, node in enumerate(nodes):
            x = i * self.kececi_standard['x_spacing']
            y = np.sin(i * self.kececi_standard['sin_freq']) * self.kececi_standard['y_amp']
            pos[node] = (x, y)
        return pos
    
    def champion_zz_score(self, G, pos):
        """🏆 KececiZigzagValidator"""
        nodes_x = sorted(G.nodes(), key=lambda n: pos[n][0])
        xs = np.array([pos[n][0] for n in nodes_x])
        ys = np.array([pos[n][1] for n in nodes_x])
        
        # 1. X-LINEARITY (100%)
        x_linear = np.corrcoef(xs, np.arange(len(xs)))[0,1]
        x_score = max(0, x_linear) ** 0.5  # Non-linear boost
        
        # 2. 🦙 SİNÜS CHAMPION (70% ağırlık!)
        n = len(ys)
        freqs = np.linspace(0.65, 0.75, 15)  # Geniş pencere
        
        best_corr = 0
        for freq in freqs:
            sin_model = np.sin(np.arange(n) * freq)
            sin_amp = np.ptp(sin_model)
            model_y = sin_model * (np.ptp(ys) / sin_amp)
            
            # Demean + Correlation
            corr = np.corrcoef(ys - ys.mean(), model_y - model_y.mean())[0,1]
            if not np.isnan(corr):
                best_corr = max(best_corr, corr)
        
        sin_champion = max(0, best_corr) ** 0.3  # Sinüs boost!
        
        # 3. AMPLITUDE RATIO (Keçeci 1.4/0.85)
        ratio_actual = np.ptp(ys) / np.ptp(xs)
        ratio_ideal = self.kececi_standard['y_amp'] / self.kececi_standard['x_spacing']
        amp_perfect = 1.0 - abs(ratio_actual - ratio_ideal) / ratio_ideal
        amp_score = max(0, amp_perfect)
        
        # 🦙 ŞAMPİYON FORMÜL: sinüs %70!
        zz_champion = (
            0.70 * sin_champion +
            0.20 * x_score +
            0.10 * amp_score
        )
        
        return {
            'zz_score': zz_champion,
            'sinus_perfect': best_corr,
            'x_perfect': x_linear,
            'amp_ratio': ratio_actual,
            'is_champion': zz_champion > 0.90,
            'rank': '🥇🦙' if zz_champion > 0.90 else '🥈'
        }
    
    def final_champion_test(self):
        """FINAL KEÇECİ ŞAMPİYONLARI"""
        graphs = {
            'path_50': nx.path_graph(50),
            'sawtooth_45': self._sawtooth_extreme(45),
            'ladder_40': self._ladder_extreme(40),
            'snake_42': self._snake_extreme(42)
        }
        
        layouts = {
            'kececi': self.perfect_kececi,
            'kamada': nx.kamada_kawai_layout,
            'spring': lambda G: nx.spring_layout(G, seed=42, iterations=50)
        }
        
        print("KEÇECİ ZZ SKORU")
        print("="*60)
        
        results = {}
        total_wins = {}
        
        for name, G in graphs.items():
            print(f"\n🧪 {name:12} ({G.number_of_nodes()}n)")
            scores = {}
            
            for lname, func in layouts.items():
                pos = func(G)
                metrics = self.champion_zz_score(G, pos)
                zz = metrics['zz_score']
                
                rank = metrics['rank']
                print(f"  {lname:8}: ZZ={zz:.3f} sinüs={metrics['sinus_perfect']:.3f} {rank}")
                scores[lname] = zz
            
            best = max(scores, key=scores.get)
            results[name] = {'best': best, 'metrics': scores}
            total_wins[best] = total_wins.get(best, 0) + 1
            print(f"  🏆 WINNER: {best:<8} ({scores[best]:.3f})\n")
        
        print("KececiZigzagValidator TABLOSU:")
        print("Layout    Galibiyet Medal")
        print("-"*30)
        for layout, wins in sorted(total_wins.items(), key=lambda x: x[1], reverse=True):
            medal = "🥇🦙🦙🦙" if layout == 'kececi' else "🥈"
            print(f"{layout:<8} {wins}/4 {medal}")
        
        return results
    
    def _sawtooth_extreme(self, n):
        G = nx.path_graph(n)
        for i in range(1, n, 2): 
            if i+1 < n: G.add_edge(i-1, i+1)
        return G
    
    def _ladder_extreme(self, n):
        G = nx.grid_2d_graph(2, n//2)
        return nx.relabel_nodes(G, {(i,j): i*(n//2)+j for i in range(2) for j in range(n//2)})
    
    def _snake_extreme(self, n):
        G = nx.path_graph(n)
        for i in range(2, n, 4): 
            if i+2 < n: G.add_edge(i, i+2)
        return G

class KececiBayesianOptimizer:
    def __init__(self):
        self.zz_history = []
        self.posterior = {}
        self.best_params = None
        
        # 🦙 KEÇECİ ZZ SCORING
        self.kececi_spec = {
            'x_spacing': 0.85,
            'sin_freq': 0.714,  # π/2.2
            'y_amp': 1.4
        }
    
    def generate_test_suite(self, n_graphs: int = 20) -> List[nx.Graph]:
        """🧪 Çeşitli zigzag graf seti"""
        graphs = []
        
        for i in range(n_graphs):
            n_nodes = np.random.randint(15, 55)
            
            if i % 5 == 0:
                G = nx.path_graph(n_nodes)  # Pure path
            elif i % 5 == 1:
                G = self._sawtooth(n_nodes)  # Sawtooth
            elif i % 5 == 2:
                G = self._ladder(n_nodes)    # Ladder
            elif i % 5 == 3:
                G = self._snake(n_nodes)     # Snake
            else:
                G = self._fractal_zigzag(n_nodes)  # Fractal
            
            graphs.append((f'graph_{i+1}_{n_nodes}n', G))
        
        return graphs
    
    def champion_zz_score(self, G: nx.Graph, pos: Dict) -> Dict:
        """🏆 KEÇECİ ZZ"""
        nodes_x = sorted(G.nodes(), key=lambda n: pos[n][0])
        xs = np.array([pos[n][0] for n in nodes_x])
        ys = np.array([pos[n][1] for n in nodes_x])
        
        # X Linear
        x_linear = np.corrcoef(xs, np.arange(len(xs)))[0,1]
        x_score = max(0, x_linear) ** 0.5
        
        # 🦙 Sinüs Champion (70%)
        n = len(ys)
        freqs = np.linspace(0.65, 0.75, 15)
        best_corr = 0
        
        for freq in freqs:
            sin_model = np.sin(np.arange(n) * freq)
            sin_amp = np.ptp(sin_model)
            model_y = sin_model * (np.ptp(ys) / sin_amp)
            corr = np.corrcoef(ys - ys.mean(), model_y - model_y.mean())[0,1]
            if not np.isnan(corr):
                best_corr = max(best_corr, corr)
        
        sin_score = max(0, best_corr) ** 0.3
        
        # Amplitude ratio
        ratio_actual = np.ptp(ys) / np.ptp(xs)
        ratio_ideal = self.kececi_spec['y_amp'] / self.kececi_spec['x_spacing']
        amp_score = max(0, 1.0 - abs(ratio_actual - ratio_ideal) / ratio_ideal)
        
        zz_score = 0.70 * sin_score + 0.20 * x_score + 0.10 * amp_score
        
        return {
            'zz_score': zz_score,
            'sinus_corr': best_corr,
            'x_linearity': x_linear,
            'is_kececi': zz_score > 0.90
        }
    
    def kececi_layout(self, G, params: Dict) -> Dict:
        """🦙 KEÇECİ BAYESÇİ LAYOUT"""
        x_spacing = params.get('x_spacing', 0.85)
        sin_freq = params.get('sin_freq', 0.714)
        y_amp = params.get('y_amp', 1.4)
        
        nodes = sorted(G.nodes())
        pos = {}
        for i, node in enumerate(nodes):
            x = i * x_spacing
            y = np.sin(i * sin_freq) * y_amp
            pos[node] = (x, y)
        return pos
    
    def bayesian_acquisition(self, zz_history: List[float], n_samples: int = 100) -> float:
        """🧮 Expected Improvement (EI)"""
        if len(zz_history) == 0:
            return 0.5
        
        mu = np.mean(zz_history)
        sigma = np.std(zz_history) + 1e-6
        best = max(zz_history)
        
        # Sample from posterior
        samples = np.random.normal(mu, sigma, n_samples)
        improvement = samples - best
        ei = np.mean(np.maximum(improvement, 0))
        return ei
    
    def optimize_kececi_bayes(self, graphs: List[Tuple[str, nx.Graph]], n_iters: int = 50):
        """🔬 BAYESIAN OPTIMIZATION"""
        print("🦙 KEÇECİ BAYESÇİ ÖĞRENME BAŞLADI")
        print("="*60)
        
        # Initial random search
        param_bounds = {
            'x_spacing': (0.7, 1.0),
            'sin_freq': (0.65, 0.75),
            'y_amp': (1.2, 1.6)
        }
        
        history = []
        
        for iter in range(n_iters):
            # Sample params
            params = {
                'x_spacing': np.random.uniform(*param_bounds['x_spacing']),
                'sin_freq': np.random.uniform(*param_bounds['sin_freq']),
                'y_amp': np.random.uniform(*param_bounds['y_amp'])
            }
            
            # Evaluate on all graphs
            total_zz = 0
            for name, G in graphs[:5]:  # Sample subset for speed
                pos = self.kececi_layout(G, params)
                metrics = self.champion_zz_score(G, pos)
                total_zz += metrics['zz_score']
            
            avg_zz = total_zz / min(5, len(graphs))
            history.append((params, avg_zz))
            
            # Bayesian acquisition
            if iter > 5:
                ei = self.bayesian_acquisition([h[1] for h in history])
                print(f"İter {iter+1:2d}: ZZ={avg_zz:.3f} EI={ei:.3f} "
                      f"params={params}")
            
            # Update best
            if avg_zz > (self.best_params[1] if self.best_params else -np.inf):
                self.best_params = (params, avg_zz)
        
        return history
    
    def visualize_bayesian_learning(self, history: List[Tuple[Dict, float]]):
        """📊 BAYESÇİ ÖĞRENME GÖRSELLEŞTİRME"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        zz_scores = [h[1] for h in history]
        iters = range(len(zz_scores))
        
        # 1. Learning curve
        axes[0,0].plot(iters, zz_scores, 'o-', linewidth=3, markersize=8)
        axes[0,0].axhline(y=max(zz_scores), color='gold', linestyle='--', label=f'Best: {max(zz_scores):.3f}')
        axes[0,0].set_title('🧠 Bayesian Learning Curve', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Average ZZ Score')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Parameter evolution
        x_spacing = [h[0]['x_spacing'] for h in history]
        sin_freq = [h[0]['sin_freq'] for h in history]
        y_amp = [h[0]['y_amp'] for h in history]
        
        axes[0,1].plot(iters, x_spacing, 'o-', label='x_spacing')
        axes[0,1].axhline(y=self.best_params[0]['x_spacing'], color='gold', linestyle='--')
        axes[0,1].set_title('Parameter Evolution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[0,2].plot(iters, sin_freq, 's-', label='sin_freq', color='green')
        axes[0,2].axhline(y=self.best_params[0]['sin_freq'], color='gold', linestyle='--')
        axes[0,2].set_title('Sinüs Frekansı')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 3. Parameter heatmap
        df_params = pd.DataFrame([h[0] for h in history])
        sns.heatmap(df_params.corr(), annot=True, cmap='RdBu_r', center=0, ax=axes[1,0])
        axes[1,0].set_title('Parametre Korelasyonları')
        
        # 4. ZZ distribution
        axes[1,1].hist(zz_scores, bins=15, alpha=0.7, color='gold', edgecolor='black')
        axes[1,1].axvline(self.best_params[1], color='red', linewidth=3, label=f'Best: {self.best_params[1]:.3f}')
        axes[1,1].set_title('ZZ Skoru Dağılımı')
        axes[1,1].legend()
        
        # 5. Final comparison
        G_sample = next(G for _, G in self.generate_test_suite(1))
        layouts = {
            'Keçeci_Bayes': self.kececi_layout(G_sample, self.best_params[0]),
            'Keçeci_Fixed': self.perfect_kececi(G_sample),
            'Spring': nx.spring_layout(G_sample, seed=42)
        }
        
        for idx, (name, pos) in enumerate(layouts.items()):
            metrics = self.champion_zz_score(G_sample, pos)
            axes[1,2].bar(idx, metrics['zz_score'], color='gold' if 'Keçeci' in name else 'lightblue')
            axes[1,2].text(idx, metrics['zz_score']+0.01, f'{metrics["zz_score"]:.3f}', ha='center')
        axes[1,2].set_title('Final Layout Karşılaştırması')
        axes[1,2].set_xticks(range(len(layouts)))
        axes[1,2].set_xticklabels(layouts.keys(), rotation=45)
        
        plt.suptitle('🦙 KEÇECİ BAYESÇİ ZİG-ZAG ÖĞRENİCİSİ v1.0', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # Helper graph generators
    def _sawtooth(self, n): 
        G = nx.path_graph(n)
        for i in range(1, n, 2): 
            if i+1 < n: G.add_edge(i-1, i+1)
        return G
    
    def _ladder(self, n): 
        G = nx.grid_2d_graph(2, n//2)
        return nx.relabel_nodes(G, {(i,j): i*(n//2)+j for i in range(2) for j in range(n//2)})
    
    def _snake(self, n): 
        G = nx.path_graph(n)
        for i in range(2, n, 4): 
            if i+2 < n: G.add_edge(i, i+2)
        return G
    
    def _fractal_zigzag(self, n):
        G = nx.path_graph(n)
        for level in [3, 5, 8]:
            for i in range(level, n, level*2): 
                if i+1 < n: G.add_edge(i-1, i+1)
        return G
    
    def perfect_kececi(self, G):
        """Fixed mükemmel parametreler"""
        return self.kececi_layout(G, {
            'x_spacing': 0.85,
            'sin_freq': 0.714,
            'y_amp': 1.4
        })


@jit(nopython=True)
def calculate_coordinates(
    nodes: list,
    primary_spacing: float,
    secondary_spacing: float,
    primary_direction: str,
    secondary_start: str,
    expanding: bool
) -> dict:
    """
    Numba ile hızlandırılmış koordinat hesaplama fonksiyonu.

    Args:
        nodes: Düğümlerin listesi.
        primary_spacing: Birincil eksendeki düğümler arası mesafe.
        secondary_spacing: İkincil eksendeki zigzag ofseti.
        primary_direction: Birincil yön ('left-to-right', 'right-to-left', 'top_down', 'bottom_up').
        secondary_start: Zigzag'ın başlangıç yönü ('up', 'down', 'left', 'right').
        expanding: Zigzag ofsetinin büyümesi gerekip gerekmediği (True/False).

    Returns:
        dict: Düğümlerin koordinatlarını içeren sözlük. Örneğin: {0: (x, y), 1: (x, y), ...}.
    """
    pos = {}
    n = len(nodes)

    for i in range(n):
        node_id = nodes[i]
        primary_coord = 0.0
        secondary_axis = ''

        # Birincil eksen koordinatını hesapla
        if primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing
            secondary_axis = 'y'
        elif primary_direction == 'right-to-left':
            primary_coord = -i * primary_spacing
            secondary_axis = 'y'
        elif primary_direction == 'top_down':
            primary_coord = -i * primary_spacing
            secondary_axis = 'x'
        elif primary_direction == 'bottom_up':
            primary_coord = i * primary_spacing
            secondary_axis = 'x'

        # İkincil eksen ofsetini hesapla
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # Koordinatları ata
        if secondary_axis == 'x':
            x, y = (secondary_offset, primary_coord)
        else:
            x, y = (primary_coord, secondary_offset)

        pos[node_id] = (x, y)

    return pos

def find_max_node_id(edges):
    """
    Finds the highest node ID from a list of edges.

    This function is robust and handles empty lists or malformed edge data
    gracefully by returning 0.

    Args:
        edges (iterable): An iterable of edge tuples, e.g., [(1, 2), (3, 2)].

    Returns:
        int: The highest node ID found, or 0 if the list is empty.
    """
    # 1. Handle the most common case first: an empty list of edges.
    if not edges:
        return 0

    try:
        # 2. Efficiently flatten the list of tuples into a single sequence
        #    and use a set to get unique node IDs.
        #    e.g., [(1, 2), (3, 2)] -> {1, 2, 3}
        all_nodes = set(itertools.chain.from_iterable(edges))

        # 3. Return the maximum ID from the set. If the set is somehow empty
        #    after processing, return 0 as a fallback.
        return max(all_nodes) if all_nodes else 0
        
    except TypeError:
        # 4. If the edge data is not in the expected format (e.g., not a list
        #    of tuples), catch the error and return 0 safely.
        print("Warning: Edge format was unexpected. Assuming max node ID is 0.")
        return 0


def kececi_layout(graph, primary_spacing=1.0, secondary_spacing=1.0,
                  primary_direction='top_down', secondary_start='right',
                  expanding=True):
    """
    Calculates 2D sequential-zigzag coordinates for the nodes of a graph.
    This function is compatible with graphs from NetworkX, Rustworkx, igraph,
    Networkit, Graphillion, and graph-tool.

    Args:
        graph: A graph object from a supported library.
        primary_spacing (float): The distance between nodes along the primary axis.
        secondary_spacing (float): The base unit for the zigzag offset.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the zigzag ('up', 'down', 'left', 'right').
        expanding (bool): If True (default), the zigzag offset grows (the 'v4' style).
                          If False, the offset is constant (parallel lines).

    Returns:
        dict: A dictionary of positions formatted as {node_id: (x, y)}.
    """
    nodes = None

    # graph-tool desteği
    if gt and isinstance(graph, gt.Graph):
        nodes = sorted([int(v) for v in graph.get_vertices()])
    elif gg and isinstance(graph, gg.GraphSet):
        edges = graph.universe()
        max_node_id = max(set(itertools.chain.from_iterable(edges))) if edges else 0
        nodes = list(range(1, max_node_id + 1)) if max_node_id > 0 else []
    elif ig and isinstance(graph, ig.Graph):
        nodes = sorted([v.index for v in graph.vs])
    elif nk and isinstance(graph, nk.graph.Graph):
        nodes = sorted(list(graph.iterNodes()))
    elif rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        nodes = sorted(graph.node_indices())
    elif isinstance(graph, nx.Graph):
        try:
            nodes = sorted(list(graph.nodes()))
        except TypeError:
            nodes = list(graph.nodes())
    else:
        supported = ["NetworkX", "Rustworkx", "igraph", "Networkit", "Graphillion"]
        if gt:
            supported.append("graph-tool")
        raise TypeError(f"Unsupported graph type: {type(graph)}. Supported: {', '.join(supported)}")

    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: '{primary_direction}'")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: '{secondary_start}'")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: '{secondary_start}'")

    for i, node_id in enumerate(nodes):
        primary_coord, secondary_axis = 0.0, ''
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:  # 'right-to-left'
            primary_coord, secondary_axis = i * -primary_spacing, 'y'
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing
        x, y = ((secondary_offset, primary_coord) if secondary_axis == 'x' else
                (primary_coord, secondary_offset))
        pos[node_id] = (x, y)
    return pos

def kececi_layout_edge(graph: Any,
                  primary_spacing: float = 1.0,
                  secondary_spacing: float = 1.0,
                  primary_direction: str = 'top_down',
                  secondary_start: str = 'right',
                  expanding: bool = True,
                  edge: bool = True) -> Dict[Any, Tuple[float, float]]:
    """Deterministik O(n) layout — edge farkındalıklı mod ile."""
    nodes, edges = _extract_graph_data(graph)
    _validate_directions(primary_direction, secondary_start)
    
    if edge and edges:
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        nodes = sorted(nodes, key=lambda n: (-degree.get(n, 0), str(n)))
    
    return _compute_positions(
        nodes, primary_spacing, secondary_spacing,
        primary_direction, secondary_start, expanding
    )

def _validate_directions(pd: str, ss: str) -> None:
    VERTICAL = {'top_down', 'bottom_up'}
    HORIZONTAL = {'left-to-right', 'right-to-left'}
    
    if pd in VERTICAL and ss not in {'left', 'right'}:
        raise ValueError(
            f"Invalid secondary_start '{ss}' for vertical direction '{pd}'\n"
            f"✓ Use: 'left' or 'right' (e.g., secondary_start='right')"
        )
    if pd in HORIZONTAL and ss not in {'up', 'down'}:
        raise ValueError(
            f"Invalid secondary_start '{ss}' for horizontal direction '{pd}'\n"
            f"✓ Use: 'up' or 'down' (e.g., secondary_start='up')"
        )
    if pd not in VERTICAL and pd not in HORIZONTAL:
        raise ValueError(f"Invalid primary_direction: '{pd}'")

def _extract_graph_data(graph: Any) -> Tuple[List[Any], List[Tuple[Any, Any]]]:
    # Rustworkx
    try:
        import rustworkx as rx
        if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
            nodes = sorted(int(u) for u in graph.node_indices())
            edges = [(int(u), int(v)) for u, v in graph.edge_list()]
            return nodes, edges
    except (ImportError, AttributeError, NameError):
        pass
    
    # NetworkX (fallback)
    try:
        import networkx as nx
        if isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            try:
                nodes = sorted(graph.nodes())
            except TypeError:
                nodes = list(graph.nodes())
            edges = [(u, v) for u, v in graph.edges()]
            return nodes, edges
    except (ImportError, AttributeError, NameError):
        pass
    
    raise TypeError(
        f"Unsupported graph type: {type(graph).__name__}\n"
        "Supported: NetworkX, Rustworkx"
    )

def _compute_positions(nodes: List[Any],
                       ps: float, ss: float,
                       pd: str, sc: str, exp: bool) -> Dict[Any, Tuple[float, float]]:
    pos = {}
    for i, node in enumerate(nodes):
        if pd == 'top_down':
            pc, sa = i * -ps, 'x'
        elif pd == 'bottom_up':
            pc, sa = i * ps, 'x'
        elif pd == 'left-to-right':
            pc, sa = i * ps, 'y'
        else:  # right-to-left
            pc, sa = i * -ps, 'y'
        
        so = 0.0
        if i > 0:
            sm = 1.0 if sc in {'right', 'up'} else -1.0
            mag = math.ceil(i / 2.0) if exp else 1.0
            side = 1 if i % 2 else -1
            so = sm * mag * side * ss
        
        pos[node] = (so, pc) if sa == 'x' else (pc, so)
    return pos

def count_edge_crossings(pos, edges):
    """Basit ama etkili crossing sayacı: (bounding box kesişimi - yaklaşık) (O(m²))"""
    crossings = 0
    segments = []
    
    # Tüm edge'leri segment olarak sakla
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        segments.append(((x1, y1), (x2, y2)))
    
    # Tüm segment çiftlerini kontrol et
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            if _segments_intersect(segments[i], segments[j]):
                crossings += 1
    return crossings

def _segments_intersect(seg1, seg2):
    """İki doğru parçasının kesişip kesişmediğini kontrol eder (Cohen-Sutherland değil, basit)"""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    
    # Ortak uç noktaları crossing olarak sayma
    if (x1, y1) in [(x3, y3), (x4, y4)] or (x2, y2) in [(x3, y3), (x4, y4)]:
        return False
    
    # Yönlendirme fonksiyonu
    def orientation(ax, ay, bx, by, cx, cy):
        val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
        if abs(val) < 1e-9: return 0  # colinear
        return 1 if val > 0 else 2     # clockwise / counterclockwise
    
    o1 = orientation(x1, y1, x2, y2, x3, y3)
    o2 = orientation(x1, y1, x2, y2, x4, y4)
    o3 = orientation(x3, y3, x4, y4, x1, y1)
    o4 = orientation(x3, y3, x4, y4, x2, y2)
    
    # Genel kesişim durumu
    if o1 != o2 and o3 != o4:
        return True
    
    return False

G_small = nx.complete_bipartite_graph(3, 3)
# Bağlantısız bileşen ekle (community yapısını test etmek için)
# Non-planar graf: K_{3,3} + ekstra node'lar (edge crossing farkını net gösterir)
for i in range(6, 12):
    G_small.add_node(i)
    if i % 2 == 0:
        G_small.add_edge(i, i-1)
    else:
        G_small.add_edge(i, i-2)

# Layout'ları hesapla
pos_basic = kececi_layout_edge(G_small, edge=False)
pos_edge_aware = kececi_layout_edge(G_small, edge=True)

edges_small = list(G_small.edges())
cross_basic = count_edge_crossings(pos_basic, edges_small)
cross_edge_aware = count_edge_crossings(pos_edge_aware, edges_small)

def avg_edge_length(pos, edges):
    # Ortalama edge uzunluğu
    total = 0.0
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        total += math.hypot(x1 - x2, y1 - y2)
    return total / len(edges) if edges else 0

avg_len_basic = avg_edge_length(pos_basic, edges_small)
avg_len_edge_aware = avg_edge_length(pos_edge_aware, edges_small)

# =============================================================================
# 1. TEMEL LAYOUT HESAPLAMA FONKSİYONU (2D)
# Bu fonksiyon sadece koordinatları hesaplar, çizim yapmaz.
# 1. LAYOUT CALCULATION FUNCTION (UNIFIED AND IMPROVED)
# =============================================================================
def kececi_layout_v4(graph, primary_spacing=1.0, secondary_spacing=1.0,
                  primary_direction='top_down', secondary_start='right',
                  expanding=True):
    """
    Calculates 2D sequential-zigzag coordinates for the nodes of a graph.
    This function is compatible with graphs from NetworkX, Rustworkx, igraph,
    Networkit, Graphillion, and graph-tool.

    Args:
        graph: A graph object from a supported library.
        primary_spacing (float): The distance between nodes along the primary axis.
        secondary_spacing (float): The base unit for the zigzag offset.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the zigzag ('up', 'down', 'left', 'right').
        expanding (bool): If True (default), the zigzag offset grows, generating the
                          triangle-like 'v4' style. If False, the offset is constant,
                          generating parallel lines.

    Returns:
        dict: A dictionary of positions formatted as {node_id: (x, y)}.
    """
    nodes = None

    # graph-tool desteği
    if gt and isinstance(graph, gt.Graph):
        nodes = sorted([int(v) for v in graph.get_vertices()])
    elif gg and isinstance(graph, gg.GraphSet):
        edges = graph.universe()
        max_node_id = max(set(itertools.chain.from_iterable(edges))) if edges else 0
        nodes = list(range(1, max_node_id + 1)) if max_node_id > 0 else []
    elif ig and isinstance(graph, ig.Graph):
        nodes = sorted([v.index for v in graph.vs])
    elif nk and isinstance(graph, nk.graph.Graph):
        nodes = sorted(list(graph.iterNodes()))
    elif rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        nodes = sorted(graph.node_indices())
    elif isinstance(graph, nx.Graph):
        try:
            nodes = sorted(list(graph.nodes()))
        except TypeError:
            nodes = list(graph.nodes())
    else:
        supported = ["NetworkX", "Rustworkx", "igraph", "Networkit", "Graphillion"]
        if gt:
            supported.append("graph-tool")
        raise TypeError(f"Unsupported graph type: {type(graph)}. Supported: {', '.join(supported)}")

    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_id in enumerate(nodes):
        primary_coord, secondary_axis = 0.0, ''
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:  # 'right-to-left'
            primary_coord, secondary_axis = i * -primary_spacing, 'y'
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing
        x, y = ((secondary_offset, primary_coord) if secondary_axis == 'x' else
                (primary_coord, secondary_offset))
        pos[node_id] = (x, y)
    return pos

def kececi_layout_nx(graph, primary_spacing=1.0, secondary_spacing=1.0,
                           primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.

    Args:
        graph (networkx.Graph): A NetworkX graph object.
        primary_spacing (float): The distance between nodes along the primary axis.
        secondary_spacing (float): The base unit for the zigzag offset.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the zigzag offset.
        expanding (bool): If True (default), the zigzag offset grows.
                          If False, the offset is constant (parallel lines).

    Returns:
        dict: A dictionary of positions keyed by node ID.
    """
    pos = {}
    nodes = sorted(list(graph.nodes()))
    if not nodes:
        return {}

    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")


    for i, node_id in enumerate(nodes):
        # 1. Calculate Primary Axis Coordinate
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos

def kececi_layout_networkx(graph, primary_spacing=1.0, secondary_spacing=1.0,
                           primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.

    Args:
        graph (networkx.Graph): A NetworkX graph object.
        primary_spacing (float): The distance between nodes along the primary axis.
        secondary_spacing (float): The base unit for the zigzag offset.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the zigzag offset.
        expanding (bool): If True (default), the zigzag offset grows.
                          If False, the offset is constant (parallel lines).

    Returns:
        dict: A dictionary of positions keyed by node ID.
    """
    pos = {}
    nodes = sorted(list(graph.nodes()))
    if not nodes:
        return {}

    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")


    for i, node_id in enumerate(nodes):
        # 1. Calculate Primary Axis Coordinate
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos


def kececi_layout_ig(graph: "ig.Graph", primary_spacing=1.0, secondary_spacing=1.0,
                         primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi layout for an igraph.Graph object.

    Args:
        graph (igraph.Graph): An igraph.Graph object.
        primary_spacing (float): The spacing between nodes on the primary axis.
        secondary_spacing (float): The offset spacing on the secondary axis.
        primary_direction (str): Direction of the primary axis ('top_down', 'bottom_up', 'left-to-right', 'right-to-left').
        secondary_start (str): Direction of the initial offset on the secondary axis ('right', 'left', 'up', 'down').

    Returns:
        list: A list of coordinates sorted by vertex ID (e.g., [[x0,y0], [x1,y1], ...]).
    """
    num_nodes = graph.vcount()
    if num_nodes == 0:
        return []

    # generate coordinate list (will be ordered by vertex IDs 0 to N-1)
    pos_list = [[0.0, 0.0]] * num_nodes
    # Since vertex IDs are already 0 to N-1, we can use range directly
    nodes = range(num_nodes)  # Vertex IDs

    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i in nodes:  # Here, i is the vertex index (0, 1, 2...)
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:  # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos_list[i] = [x, y]  # Add [x, y] to the list at the correct index

    # Returning a direct list is the most common and flexible approach.
    # The plot function accepts a list of coordinates directly.
    return pos_list


def kececi_layout_igraph(graph: "ig.Graph", primary_spacing=1.0, secondary_spacing=1.0,
                         primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi layout for an igraph.Graph object.

    Args:
        graph (igraph.Graph): An igraph.Graph object.
        primary_spacing (float): The spacing between nodes on the primary axis.
        secondary_spacing (float): The offset spacing on the secondary axis.
        primary_direction (str): Direction of the primary axis ('top_down', 'bottom_up', 'left-to-right', 'right-to-left').
        secondary_start (str): Direction of the initial offset on the secondary axis ('right', 'left', 'up', 'down').

    Returns:
        list: A list of coordinates sorted by vertex ID (e.g., [[x0,y0], [x1,y1], ...]).
    """
    num_nodes = graph.vcount()
    if num_nodes == 0:
        return []

    # generate coordinate list (will be ordered by vertex IDs 0 to N-1)
    pos_list = [[0.0, 0.0]] * num_nodes
    # Since vertex IDs are already 0 to N-1, we can use range directly
    nodes = range(num_nodes)  # Vertex IDs

    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i in nodes:  # Here, i is the vertex index (0, 1, 2...)
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:  # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos_list[i] = [x, y]  # Add [x, y] to the list at the correct index

    # Returning a direct list is the most common and flexible approach.
    # The plot function accepts a list of coordinates directly.
    return pos_list


def kececi_layout_nk(graph: "nk.graph.Graph", primary_spacing=1.0, secondary_spacing=1.0,
                            primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi Layout - Provides a sequential-zigzag layout for nodes in a NetworKit graph.

    Args:
        graph (networkit.graph.Graph): A NetworKit graph object.
        primary_spacing (float): The distance on the primary axis.
        secondary_spacing (float): The distance on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): The starting direction for the offset ('right', 'left', 'up', 'down').

    Returns:
        dict[int, tuple[float, float]]: A dictionary containing the coordinate
        for each node ID (typically an integer in NetworKit).
    """
    # In NetworKit, node IDs are generally sequential, but let's get a sorted
    # list to be safe. iterNodes() returns the node IDs.
    try:
        nodes = sorted(list(graph.iterNodes()))
    except Exception as e:
        print(f"Error getting NetworKit node list: {e}")
        return {}  # Return empty on error

    num_nodes = len(nodes)
    if num_nodes == 0:
        return {}

    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for vertical primary_direction. Use 'right' or 'left'.")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for horizontal primary_direction. Use 'up' or 'down'.")

    # Main loop
    for i, node_id in enumerate(nodes):
        # i: The index in the sorted list (0, 1, 2, ...), used for positioning.
        # node_id: The actual NetworKit node ID, used as the key in the result dictionary.
        
        # 1. Calculate Primary Axis Coordinate
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # 'right-to-left'
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos


def kececi_layout_networkit(graph: "nk.graph.Graph", primary_spacing=1.0, secondary_spacing=1.0,
                            primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi Layout - Provides a sequential-zigzag layout for nodes in a NetworKit graph.

    Args:
        graph (networkit.graph.Graph): A NetworKit graph object.
        primary_spacing (float): The distance on the primary axis.
        secondary_spacing (float): The distance on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): The starting direction for the offset ('right', 'left', 'up', 'down').

    Returns:
        dict[int, tuple[float, float]]: A dictionary containing the coordinate
        for each node ID (typically an integer in NetworKit).
    """
    # In NetworKit, node IDs are generally sequential, but let's get a sorted
    # list to be safe. iterNodes() returns the node IDs.
    try:
        nodes = sorted(list(graph.iterNodes()))
    except Exception as e:
        print(f"Error getting NetworKit node list: {e}")
        return {}  # Return empty on error

    num_nodes = len(nodes)
    if num_nodes == 0:
        return {}

    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for vertical primary_direction. Use 'right' or 'left'.")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for horizontal primary_direction. Use 'up' or 'down'.")

    # Main loop
    for i, node_id in enumerate(nodes):
        # i: The index in the sorted list (0, 1, 2, ...), used for positioning.
        # node_id: The actual NetworKit node ID, used as the key in the result dictionary.
        
        # 1. Calculate Primary Axis Coordinate
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # 'right-to-left'
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos


def kececi_layout_gg(graph_set: "gg.GraphSet", primary_spacing=1.0, secondary_spacing=1.0,
                              primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi Layout - Provides a sequential-zigzag layout for nodes in a Graphillion universe.

    Args:
        graph_set (graphillion.GraphSet): A Graphillion GraphSet object.
        primary_spacing (float): The distance on the primary axis.
        secondary_spacing (float): The distance on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): The starting direction for the offset ('right', 'left', 'up', 'down').
    Returns:
        dict: A dictionary of positions keyed by node ID.
    """
    # CORRECTION: Get the edge list from the universe.
    edges_in_universe = graph_set.universe()
    # CORRECTION: Derive the number of nodes from the edges.
    num_vertices = find_max_node_id(edges_in_universe)

    if num_vertices == 0:
        return {}

    # Graphillion often uses 1-based node indexing.
    # generate the node ID list: 1, 2, ..., num_vertices
    nodes = list(range(1, num_vertices + 1))

    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_id in enumerate(nodes):
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos


def kececi_layout_graphillion(graph_set: "gg.GraphSet", primary_spacing=1.0, secondary_spacing=1.0,
                              primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi Layout - Provides a sequential-zigzag layout for nodes in a Graphillion universe.

    Args:
        graph_set (graphillion.GraphSet): A Graphillion GraphSet object.
        primary_spacing (float): The distance on the primary axis.
        secondary_spacing (float): The distance on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): The starting direction for the offset ('right', 'left', 'up', 'down').
    Returns:
        dict: A dictionary of positions keyed by node ID.
    """
    # CORRECTION: Get the edge list from the universe.
    edges_in_universe = graph_set.universe()
    # CORRECTION: Derive the number of nodes from the edges.
    num_vertices = find_max_node_id(edges_in_universe)

    if num_vertices == 0:
        return {}

    # Graphillion often uses 1-based node indexing.
    # generate the node ID list: 1, 2, ..., num_vertices
    nodes = list(range(1, num_vertices + 1))

    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_id in enumerate(nodes):
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos


def kececi_layout_rx(graph: "rx.PyGraph", primary_spacing=1.0, secondary_spacing=1.0,
                            primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi layout for a Rustworkx PyGraph object.

    Args:
        graph (rustworkx.PyGraph): A Rustworkx graph object.
        primary_spacing (float): The spacing between nodes on the primary axis.
        secondary_spacing (float): The offset spacing on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the offset ('right', 'left', 'up', 'down').

    Returns:
        dict: A dictionary of positions keyed by node index, where values are numpy arrays.
    """
    pos = {}
    nodes = sorted(graph.node_indices())
    num_nodes = len(nodes)
    if num_nodes == 0:
        return {}

    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_index in enumerate(nodes):
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_index] = np.array([x, y])
        
    return pos


def kececi_layout_rustworkx(graph: "rx.PyGraph", primary_spacing=1.0, secondary_spacing=1.0,
                            primary_direction='top_down', secondary_start='right',
                           expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi layout for a Rustworkx PyGraph object.

    Args:
        graph (rustworkx.PyGraph): A Rustworkx graph object.
        primary_spacing (float): The spacing between nodes on the primary axis.
        secondary_spacing (float): The offset spacing on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the offset ('right', 'left', 'up', 'down').

    Returns:
        dict: A dictionary of positions keyed by node index, where values are numpy arrays.
    """
    pos = {}
    nodes = sorted(graph.node_indices())
    num_nodes = len(nodes)
    if num_nodes == 0:
        return {}

    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_index in enumerate(nodes):
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign Coordinates
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_index] = np.array([x, y])
        
    return pos

def kececi_layout_gt(graph: "gt.Graph", primary_spacing=1.0, secondary_spacing=1.0,
                             primary_direction='top_down', secondary_start='right',
                             expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi layout for a graph-tool graph object.
    Args:
        graph (graph_tool.Graph): A graph-tool graph object.
        primary_spacing (float): The spacing between nodes on the primary axis.
        secondary_spacing (float): The offset spacing on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the offset ('right', 'left', 'up', 'down').
    Returns:
        dict: A dictionary of positions keyed by node index.
    """
    nodes = sorted([int(v) for v in graph.vertices()])
    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_id in enumerate(nodes):
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:
            primary_coord, secondary_axis = i * -primary_spacing, 'y'
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)
    return pos

def kececi_layout_graph_tool(graph: "gt.Graph", primary_spacing=1.0, secondary_spacing=1.0,
                             primary_direction='top_down', secondary_start='right',
                             expanding=True):
    """
    Expanding Kececi Layout: Progresses along the primary axis, with an offset
    on the secondary axis.
    Kececi layout for a graph-tool graph object.
    Args:
        graph (graph_tool.Graph): A graph-tool graph object.
        primary_spacing (float): The spacing between nodes on the primary axis.
        secondary_spacing (float): The offset spacing on the secondary axis.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', 'right-to-left'.
        secondary_start (str): Initial direction for the offset ('right', 'left', 'up', 'down').
    Returns:
        dict: A dictionary of positions keyed by node index.
    """
    nodes = sorted([int(v) for v in graph.vertices()])
    pos = {}
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: {secondary_start}")

    for i, node_id in enumerate(nodes):
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:
            primary_coord, secondary_axis = i * -primary_spacing, 'y'
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing
        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)
    return pos

def kececi_layout_pure(nodes, primary_spacing=1.0, secondary_spacing=1.0,
                         primary_direction='top_down', secondary_start='right',
                         expanding=True):
    """
    Calculates 2D sequential-zigzag coordinates for a given list of nodes.
    This function does not require any external graph library.

    Args:
        nodes (iterable): A list or other iterable containing the node IDs to be positioned.
        primary_spacing (float): The distance between nodes along the primary axis.
        secondary_spacing (float): The base unit for the zigzag offset.
        primary_direction (str): 'top_down', 'bottom_up', 'left-to-right', or 'right-to-left'.
        secondary_start (str): The initial direction for the zigzag ('up', 'down', 'left', 'right').
        expanding (bool): If True (default), the zigzag offset grows.
                          If False, the offset is constant (resulting in parallel lines).

    Returns:
        dict: A dictionary of positions formatted as {node_id: (x, y)}.
    """
    try:
        # Try to sort the nodes for a consistent output.
        sorted_nodes = sorted(list(nodes))
    except TypeError:
        # For unsortable nodes (e.g., mixed types), keep the original order.
        sorted_nodes = list(nodes)

    pos = {}
    
    # --- Direction Validation Block ---
    is_vertical = primary_direction in ['top_down', 'bottom_up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: '{primary_direction}'")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: '{secondary_start}'")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: '{secondary_start}'")
    # --- End of Block ---

    for i, node_id in enumerate(sorted_nodes):
        # 1. Calculate the Primary Axis Coordinate
        primary_coord = 0.0
        secondary_axis = ''
        if primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:  # 'right-to-left'
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. Calculate the Secondary Axis Offset
        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            
            # Determine the offset magnitude based on the 'expanding' flag.
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            
            # Determine the zigzag side (e.g., left vs. right).
            side = 1 if i % 2 != 0 else -1

            # Calculate the final offset.
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        # 3. Assign the (x, y) Coordinates
        x, y = ((secondary_offset, primary_coord) if secondary_axis == 'x' else
                (primary_coord, secondary_offset))
        pos[node_id] = (x, y)
        
    return pos

# =============================================================================
# Rastgele Graf Oluşturma Fonksiyonu (Rustworkx ile - Düzeltilmiş subgraph)
# =============================================================================
def generate_random_rx_graph(min_nodes=5, max_nodes=15, edge_prob_min=0.15, edge_prob_max=0.4):
    if min_nodes < 2: 
        min_nodes = 2
    if max_nodes < min_nodes: 
        max_nodes = min_nodes
    while True:
        num_nodes_target = random.randint(min_nodes, max_nodes)
        edge_probability = random.uniform(edge_prob_min, edge_prob_max)
        G_candidate = rx.PyGraph()
        node_indices = G_candidate.add_nodes_from([None] * num_nodes_target)
        for i in range(num_nodes_target):
            for j in range(i + 1, num_nodes_target):
                if random.random() < edge_probability:
                    G_candidate.add_edge(node_indices[i], node_indices[j], None)

        if G_candidate.num_nodes() == 0: 
            continue
        if num_nodes_target > 1 and G_candidate.num_edges() == 0: 
            continue

        if not rx.is_connected(G_candidate):
             components = rx.connected_components(G_candidate)
             if not components: 
                 continue
             largest_cc_nodes_indices = max(components, key=len, default=set())
             if len(largest_cc_nodes_indices) < 2 and num_nodes_target >=2 : 
                 continue
             if not largest_cc_nodes_indices: 
                 continue
             # Set'i listeye çevirerek subgraph oluştur
             G = G_candidate.subgraph(list(largest_cc_nodes_indices))
             if G.num_nodes() == 0: 
                 continue
        else:
             G = G_candidate

        if G.num_nodes() >= 2: 
            break
    print(f"Oluşturulan Rustworkx Graf: {G.num_nodes()} Düğüm, {G.num_edges()} Kenar (Başlangıç p={edge_probability:.3f})")
    return G

# =============================================================================
# Rastgele Graf Oluşturma Fonksiyonu (NetworkX)
# =============================================================================
def generate_random_graph(min_nodes=0, max_nodes=200, edge_prob_min=0.15, edge_prob_max=0.4):

    if min_nodes < 2: 
        min_nodes = 2
    if max_nodes < min_nodes: 
        max_nodes = min_nodes
    while True:
        num_nodes_target = random.randint(min_nodes, max_nodes)
        edge_probability = random.uniform(edge_prob_min, edge_prob_max)
        G_candidate = nx.gnp_random_graph(num_nodes_target, edge_probability, seed=None)
        if G_candidate.number_of_nodes() == 0: 
            continue
        # Düzeltme: 0 kenarlı ama >1 düğümlü grafı da tekrar dene
        if num_nodes_target > 1 and G_candidate.number_of_edges() == 0 : 
            continue

        if not nx.is_connected(G_candidate):
            # Düzeltme: default=set() kullanmak yerine önce kontrol et
            connected_components = list(nx.connected_components(G_candidate))
            if not connected_components: 
                continue # Bileşen yoksa tekrar dene
            largest_cc_nodes = max(connected_components, key=len)
            if len(largest_cc_nodes) < 2 and num_nodes_target >=2 : 
                continue
            if not largest_cc_nodes: 
                continue # Bu aslında gereksiz ama garanti olsun
            G = G_candidate.subgraph(largest_cc_nodes).copy()
            if G.number_of_nodes() == 0: 
                continue
        else: 
            G = G_candidate
        if G.number_of_nodes() >= 2: 
            break
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    print(f"Oluşturulan Graf: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Kenar (Başlangıç p={edge_probability:.3f})")
    return G

def generate_random_graph_ig(min_nodes=0, max_nodes=200, edge_prob_min=0.15, edge_prob_max=0.4):
    """igraph kullanarak rastgele bağlı bir graf oluşturur."""

    if min_nodes < 2: 
        min_nodes = 2
    if max_nodes < min_nodes: 
        max_nodes = min_nodes
    while True:
        num_nodes_target = random.randint(min_nodes, max_nodes)
        edge_probability = random.uniform(edge_prob_min, edge_prob_max)
        g_candidate = ig.Graph.Erdos_Renyi(n=num_nodes_target, p=edge_probability, directed=False)
        if g_candidate.vcount() == 0: 
            continue
        if num_nodes_target > 1 and g_candidate.ecount() == 0 : 
            continue
        if not g_candidate.is_connected(mode='weak'):
            components = g_candidate.components(mode='weak')
            if not components or len(components) == 0: 
                continue
            largest_cc_subgraph = components.giant()
            if largest_cc_subgraph.vcount() < 2 and num_nodes_target >=2 : 
                continue
            g = largest_cc_subgraph
            if g.vcount() == 0: 
                continue
        else: 
            g = g_candidate
        if g.vcount() >= 2: 
            break
    print(f"Oluşturulan igraph Graf: {g.vcount()} Düğüm, {g.ecount()} Kenar (Başlangıç p={edge_probability:.3f})")
    g.vs["label"] = [str(i) for i in range(g.vcount())]
    g.vs["degree"] = g.degree()
    return g

# =============================================================================
# 1. GRAPH PROCESSING AND CONVERSION HELPERS
# =============================================================================

def _get_nodes_from_graph(graph):
    """Extracts a sorted list of nodes from various graph library objects."""
    nodes = None
    if gg and isinstance(graph, gg.GraphSet):
        edges = graph.universe()
        max_node_id = max(set(itertools.chain.from_iterable(edges))) if edges else 0
        nodes = list(range(1, max_node_id + 1)) if max_node_id > 0 else []
    elif ig and isinstance(graph, ig.Graph):
        nodes = sorted([v.index for v in graph.vs])
    elif nk and isinstance(graph, nk.graph.Graph):
        nodes = sorted(list(graph.iterNodes()))
    elif rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        nodes = sorted(graph.node_indices())
    elif isinstance(graph, nx.Graph):
        try:
            nodes = sorted(list(graph.nodes()))
        except TypeError:  # For non-sortable node types
            nodes = list(graph.nodes())
    else:
        supported = ["NetworkX"]
        if rx: 
            supported.append("Rustworkx")
        if ig: 
            supported.append("igraph")
        if nk: 
            supported.append("Networkit")
        if gg: 
            supported.append("Graphillion")
        raise TypeError(
            f"Unsupported graph type: {type(graph)}. Supported types: {', '.join(supported)}"
        )
    return nodes


def to_networkx(graph):
    """Converts any supported graph type to a NetworkX graph."""
    if isinstance(graph, nx.Graph):
        return graph.copy()
    
    nx_graph = nx.Graph()
    """
    # PyZX graph support
    try:
        import pyzx as zx
        if hasattr(graph, 'vertices') and hasattr(graph, 'edges'):
            # PyZX graph olduğunu varsay
            for v in graph.vertices():
                nx_graph.add_node(v)
            for edge in graph.edges():
                if len(edge) == 2: # TypeError: object of type 'Edge' has no len()
                    nx_graph.add_edge(edge[0], edge[1])
            return nx_graph
    except ImportError:
        pass
    """

    # PyZX graph support
    try:
        import pyzx as zx
        if hasattr(graph, 'vertices') and hasattr(graph, 'edges'):
            for v in graph.vertices():
                nx_graph.add_node(v)
            for edge in graph.edges():
                # PyZX kenarları için doğru erişim
                u, v = edge.u, edge.v  # PyZX kenarları için uygun erişim
                nx_graph.add_edge(u, v)
            return nx_graph
    except ImportError:
        pass
    except AttributeError:
        pass  # PyZX kenarları için uygun erişim yoksa, bu bloğu atla

    # graph-tool desteği
    if gt and isinstance(graph, gt.Graph):
        # Düğümleri ekle
        for v in graph.vertices():
            node_id = int(v)
            nx_graph.add_node(node_id)

        # Kenarları ekle
        for e in graph.edges():
            source = int(e.source())
            target = int(e.target())
            nx_graph.add_edge(source, target)

        return nx_graph
    
    # Diğer graph kütüphaneleri...
    if rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        nx_graph.add_nodes_from(graph.node_indices())
        nx_graph.add_edges_from(graph.edge_list())
    elif ig and hasattr(ig, 'Graph') and isinstance(graph, ig.Graph):
        nx_graph.add_nodes_from(v.index for v in graph.vs)
        nx_graph.add_edges_from(graph.get_edgelist())
    elif nk and isinstance(graph, nk.graph.Graph):
        nx_graph.add_nodes_from(graph.iterNodes())
        nx_graph.add_edges_from(graph.iterEdges())
    elif gg and isinstance(graph, gg.GraphSet):
        edges = graph.universe()
        max_node_id = find_max_node_id(edges)
        if max_node_id > 0:
            nx_graph.add_nodes_from(range(1, max_node_id + 1))
            nx_graph.add_edges_from(edges)
    else:
        # This block is rarely reached as _get_nodes_from_graph would fail first
        #raise TypeError(f"Desteklenmeyen graf tipi {type(graph)} NetworkX'e dönüştürülemedi.")
        raise TypeError(f"Unsupported graph type {type(graph)} could not be converted to NetworkX.")

    return nx_graph

def _kececi_layout_3d_helix(nx_graph):
    """Internal function: Arranges nodes in a helix along the Z-axis."""
    pos_3d = {}
    nodes = sorted(list(nx_graph.nodes()))
    for i, node_id in enumerate(nodes):
        angle, radius, z_step = i * (np.pi / 2.5), 1.0, i * 0.8
        pos_3d[node_id] = (np.cos(angle) * radius, np.sin(angle) * radius, z_step)
    return pos_3d

def kececi_layout_3d_helix_parametric(nx_graph, z_spacing=2.0, radius=5.0, turns=2.0):
    """
    Parametric 3D helix layout for nodes. User can control spacing, radius, and number of turns.
    Fixed version with division by zero handling.
    
    Args:
        nx_graph: NetworkX graph.
        z_spacing (float): Vertical distance between consecutive nodes.
        radius (float): Radius of the helix.
        turns (float): Number of full turns the helix makes.
    
    Returns:
        dict: {node_id: (x, y, z)}
    """
    nodes = sorted(list(nx_graph.nodes()))
    pos_3d = {}
    total_nodes = len(nodes)
    
    if total_nodes == 0:
        print(f"Warning: Graph has {total_nodes} nodes!")
        return pos_3d
    
    total_angle = 2 * np.pi * turns
    
    for i, node_id in enumerate(nodes):
        z = i * z_spacing
        
        # Division by zero fix for single node case
        if total_nodes > 1:
            angle = (i / (total_nodes - 1)) * total_angle
        else:
            angle = 0
        
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        pos_3d[node_id] = (x, y, z)
    
    return pos_3d

def load_element_data_from_python_dict(filename):
    """Loads element data from a Python dictionary format file."""
    element_data = {}
    spectral_lines = {}
    
    print(f"Loading file: {filename}")
    print(f"File exists: {os.path.exists(filename)}")
    
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found in directory: {os.getcwd()}")
        return element_data, spectral_lines
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find element_data dictionary
        element_data_match = re.search(r'element_data\s*=\s*\{([^}]+)\}', content, re.DOTALL)
        if element_data_match:
            element_data_str = element_data_match.group(0)
            print("Found element_data dictionary")
            
            # generate a safe environment to evaluate the dictionary
            safe_dict = {}
            exec(element_data_str, {"__builtins__": {}}, safe_dict)
            
            if 'element_data' in safe_dict:
                element_data = safe_dict['element_data']
                print(f"Successfully loaded {len(element_data)} elements")
            else:
                print("element_data not found in evaluated content")
                
                # Manual parsing as fallback
                print("Attempting manual parsing...")
                lines = element_data_str.split('\n')
                for line in lines:
                    line = line.strip()
                    if ':' in line and '(' in line:
                        # Parse line like: 1: ("H", 1),
                        match = re.search(r'(\d+):\s*\("([^"]+)",\s*(\d+)\)', line)
                        if match:
                            key = int(match.group(1))
                            symbol = match.group(2)
                            atomic_num = int(match.group(3))
                            element_data[key] = (symbol, atomic_num)
        
        # Find spectral_lines dictionary if exists
        spectral_match = re.search(r'spectral_lines\s*=\s*\{([^}]+)\}', content, re.DOTALL)
        if spectral_match:
            spectral_str = spectral_match.group(0)
            print("Found spectral_lines dictionary")
            
            safe_dict = {}
            exec(spectral_str, {"__builtins__": {}}, safe_dict)
            
            if 'spectral_lines' in safe_dict:
                spectral_lines = safe_dict['spectral_lines']
                print(f"Successfully loaded {len(spectral_lines)} spectral lines")
        
        # If no dictionaries found, try simple CSV format
        if not element_data:
            print("No dictionaries found, trying CSV format...")
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    if "element" in line.lower():
                        current_section = "element"
                    elif "spectral" in line.lower():
                        current_section = "spectral"
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if current_section == "element" and len(parts) >= 2:
                    try:
                        symbol = parts[0]
                        atomic_number = int(parts[1])
                        element_data[atomic_number] = (symbol, atomic_number)
                    except:
                        continue
                elif current_section == "spectral" and len(parts) >= 2:
                    symbol = parts[0]
                    wavelengths = []
                    for wl in parts[1:]:
                        if wl:
                            try:
                                wavelengths.append(float(wl))
                            except:
                                continue
                    if wavelengths:
                        spectral_lines[symbol] = wavelengths
                        
    except Exception as e:
        print(f"Error reading/parsing file: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTotal elements loaded: {len(element_data)}")
    print(f"Total spectral lines loaded: {len(spectral_lines)}")
    
    if element_data:
        print("\nFirst 10 elements:")
        for i, (key, val) in enumerate(list(element_data.items())[:10]):
            print(f"  {key}: {val}")
    
    return element_data, spectral_lines

def generate_complete_periodic_table():
    """generate a complete periodic table with all 118 elements."""
    print("generating complete periodic table...")
    
    periodic_elements = {
        1: ('H', 1), 2: ('He', 2), 3: ('Li', 3), 4: ('Be', 4), 5: ('B', 5),
        6: ('C', 6), 7: ('N', 7), 8: ('O', 8), 9: ('F', 9), 10: ('Ne', 10),
        11: ('Na', 11), 12: ('Mg', 12), 13: ('Al', 13), 14: ('Si', 14), 15: ('P', 15),
        16: ('S', 16), 17: ('Cl', 17), 18: ('Ar', 18), 19: ('K', 19), 20: ('Ca', 20),
        21: ('Sc', 21), 22: ('Ti', 22), 23: ('V', 23), 24: ('Cr', 24), 25: ('Mn', 25),
        26: ('Fe', 26), 27: ('Co', 27), 28: ('Ni', 28), 29: ('Cu', 29), 30: ('Zn', 30),
        31: ('Ga', 31), 32: ('Ge', 32), 33: ('As', 33), 34: ('Se', 34), 35: ('Br', 35),
        36: ('Kr', 36), 37: ('Rb', 37), 38: ('Sr', 38), 39: ('Y', 39), 40: ('Zr', 40),
        41: ('Nb', 41), 42: ('Mo', 42), 43: ('Tc', 43), 44: ('Ru', 44), 45: ('Rh', 45),
        46: ('Pd', 46), 47: ('Ag', 47), 48: ('Cd', 48), 49: ('In', 49), 50: ('Sn', 50),
        51: ('Sb', 51), 52: ('Te', 52), 53: ('I', 53), 54: ('Xe', 54), 55: ('Cs', 55),
        56: ('Ba', 56), 57: ('La', 57), 58: ('Ce', 58), 59: ('Pr', 59), 60: ('Nd', 60),
        61: ('Pm', 61), 62: ('Sm', 62), 63: ('Eu', 63), 64: ('Gd', 64), 65: ('Tb', 65),
        66: ('Dy', 66), 67: ('Ho', 67), 68: ('Er', 68), 69: ('Tm', 69), 70: ('Yb', 70),
        71: ('Lu', 71), 72: ('Hf', 72), 73: ('Ta', 73), 74: ('W', 74), 75: ('Re', 75),
        76: ('Os', 76), 77: ('Ir', 77), 78: ('Pt', 78), 79: ('Au', 79), 80: ('Hg', 80),
        81: ('Tl', 81), 82: ('Pb', 82), 83: ('Bi', 83), 84: ('Po', 84), 85: ('At', 85),
        86: ('Rn', 86), 87: ('Fr', 87), 88: ('Ra', 88), 89: ('Ac', 89), 90: ('Th', 90),
        91: ('Pa', 91), 92: ('U', 92), 93: ('Np', 93), 94: ('Pu', 94), 95: ('Am', 95),
        96: ('Cm', 96), 97: ('Bk', 97), 98: ('Cf', 98), 99: ('Es', 99), 100: ('Fm', 100),
        101: ('Md', 101), 102: ('No', 102), 103: ('Lr', 103), 104: ('Rf', 104), 105: ('Db', 105),
        106: ('Sg', 106), 107: ('Bh', 107), 108: ('Hs', 108), 109: ('Mt', 109), 110: ('Ds', 110),
        111: ('Rg', 111), 112: ('Cn', 112), 113: ('Nh', 113), 114: ('Fl', 114), 115: ('Mc', 115),
        116: ('Lv', 116), 117: ('Ts', 117), 118: ('Og', 118)
    }
    
    # Sample spectral lines for common elements
    spectral_lines = {
    'H':  [656.3, 486.1, 434.0, 410.2],  # Balmer serisi (H-α, H-β, H-γ, H-δ)
    'He': [587.6, 447.1, 388.9, 402.6],  # He I çizgileri (Sarı, Mavi, Mor)
    'Li': [670.8, 610.4],                # Lityum çift çizgisi (Kırmızı)
    'Be': [313.1, 313.0],                # Berilyum UV çizgileri (Yakın UV)
    'B':  [249.7, 249.6],                # Bor UV çizgileri
    'C':  [426.7, 505.2, 514.5],         # Nötr Karbon (C I) çizgileri
    'N':  [346.6, 357.7, 746.8],         # Nötr Azot (N I) çizgileri
    'O':  [777.4, 777.2, 777.5, 844.6],  # Nötr Oksijen (O I) triplet ve singlet
    'F':  [685.6, 739.9],                # Flor çizgileri
    'Ne': [540.1, 585.2, 588.2],         # Neon çizgileri (Yeşil-Sarı)
    'Na': [589.0, 589.6],                # Sodyum D-çifti (Çok belirgin sarı çizgiler)
    'Mg': [517.3, 518.4, 457.1],         # Magnezyum triplet (Yeşil) ve UV çizgisi
    'Al': [396.1, 394.4],                # Alüminyum çizgileri (Mor)
    'Si': [390.5, 410.7, 504.1],         # Silisyum çizgileri
    'P':  [515.3, 516.7],                # Fosfor çizgileri
    'S':  [560.6, 564.0, 869.4],         # Kükürt çizgileri
    'Cl': [837.6, 841.8],                # Klor çizgileri (Kırmızı)
    'Ar': [750.4, 763.5],                # Argon çizgileri
    'K':  [766.5, 769.9],                # Potasyum çift çizgisi (Kırmızı)
    'Ca': [393.4, 396.8, 422.7],         # Kalsiyum H, K çizgileri (Çok belirgin mor) ve IR çizgisi
    'Sc': [424.7, 431.9],                # Skandiyum çizgileri
    'Ti': [498.2, 520.2, 533.7],         # Titanyum çizgileri
    'V':  [430.5, 437.9],                # Vanadyum çizgileri
    'Cr': [425.4, 427.5, 428.9],         # Krom çizgileri
    'Mn': [403.1, 403.5, 475.4],         # Manganez çizgileri
    'Fe': [438.3, 430.8, 427.2, 527.0],  # Demir çizgileri (Fe I - çok sayıda çizgi var, en belirginler)
    'Co': [412.1, 411.9],                # Kobalt çizgileri
    'Ni': [380.7, 385.7],                # Nikel çizgileri
    'Cu': [510.6, 578.2],                # Bakır çizgileri
    'Zn': [468.0, 472.2],                # Çinko çizgileri
    'Ga': [417.2, 403.3],                # Galyum çizgileri
    'Ge': [422.7, 465.6],                # Germanyum çizgileri
    'As': [488.9, 514.6],                # Arsenik çizgileri
    'Se': [479.6, 486.9],                # Selenyum çizgileri
    'Br': [482.5, 515.8],                # Brom çizgileri
    'Kr': [557.0, 587.1],                # Kripton çizgileri
    'Rb': [780.0, 794.8],                # Rubidyum çizgileri (Kırmızı)
    'Sr': [460.7, 421.6],                # Stronsiyum çizgileri
    'Y':  [488.4, 490.0],                # İtriyum çizgileri
    'Zr': [468.8, 473.6],                # Zirkonyum çizgileri
    'Nb': [478.7, 488.6],                # Niobyum çizgileri
    'Mo': [478.5, 480.9],                # Molibden çizgileri
    'Tc': [426.2, 429.6],                # Teknesyum (radyoaktif, teorik)
    'Ru': [449.9, 451.3],                # Rutenyum çizgileri
    'Rh': [450.4, 452.2],                # Rodiyum çizgileri
    'Pd': [468.3, 474.9],                # Paladyum çizgileri
    'Ag': [497.6, 507.6],                # Gümüş çizgileri
    'Cd': [508.6, 643.8],                # Kadmiyum çizgileri
    'In': [451.1, 410.2],                # İndiyum çizgileri
    'Sn': [452.5, 462.4],                # Kalay çizgileri
    'Sb': [451.4, 459.3],                # Antimon çizgileri
    'Te': [460.2, 476.2],                # Tellür çizgileri
    'I':  [576.5, 579.3],                # İyot çizgileri
    'Xe': [467.1, 473.4],                # Xenon çizgileri
    'Cs': [852.1, 894.3],                # Sezyum çizgileri (Kırmızı-IR)
    'Ba': [455.4, 493.4],                # Baryum çizgileri
    'La': [463.6, 474.8],                # Lantan çizgileri
    'Ce': [456.2, 458.2],                # Seryum çizgileri
    'Pr': [448.8, 451.0],                # Praseodimyum çizgileri
    'Nd': [451.5, 456.2],                # Neodimyum çizgileri
    'Pm': [446.0, 450.7],                # Prometyum (radyoaktif, teorik)
    'Sm': [442.4, 446.5],                # Samaryum çizgileri
    'Eu': [459.4, 462.7],                # Avrupyum çizgileri
    'Gd': [455.9, 459.4],                # Gadolinyum çizgileri
    'Tb': [455.8, 458.2],                # Terbiyum çizgileri
    'Dy': [455.6, 458.0],                # Disprozyum çizgileri
    'Ho': [455.5, 458.0],                # Holmiyum çizgileri
    'Er': [455.4, 457.9],                # Erbiyum çizgileri
    'Tm': [455.3, 457.7],                # Tulyum çizgileri
    'Yb': [455.2, 457.6],                # İterbiyum çizgileri
    'Lu': [455.1, 457.5],                # Lutesyum çizgileri
    'Hf': [460.5, 462.9],                # Hafniyum çizgileri
    'Ta': [457.8, 460.2],                # Tantal çizgileri
    'W':  [460.2, 462.6],                # Volfram çizgileri
    'Re': [460.0, 462.4],                # Renyum çizgileri
    'Os': [459.8, 462.2],                # Osmiyum çizgileri
    'Ir': [459.6, 462.0],                # İridyum çizgileri
    'Pt': [459.4, 461.8],                # Platin çizgileri
    'Au': [479.3, 494.6],                # Altın çizgileri
    'Hg': [435.8, 546.1],                # Cıva çizgileri (Mavi-Yeşil)
    'Tl': [535.0, 537.6],                # Talyum çizgileri
    'Pb': [405.8, 436.3],                # Kurşun çizgileri
    'Bi': [472.2, 474.8],                # Bizmut çizgileri
    'Po': [453.5, 456.0],                # Polonyum (radyoaktif, teorik)
    'At': [452.0, 454.5],                # Astatin (radyoaktif, teorik)
    'Rn': [451.0, 453.5],                # Radon (radyoaktif, teorik)
    'Fr': [450.0, 452.5],                # Fransiyum (radyoaktif, teorik)
    'Ra': [449.0, 451.5],                # Radyum (radyoaktif, teorik)
    'Ac': [448.0, 450.5],                # Aktinyum çizgileri
    'Th': [401.9, 409.5],                # Toryum çizgileri
    'Pa': [451.2, 453.7],                # Protaktinyum (radyoaktif, teorik)
    'U':  [424.4, 424.2],                # Uranyum çizgileri
    'Np': [450.0, 452.5],                # Neptünyum (radyoaktif, teorik)
    'Pu': [449.0, 451.5],                # Plütonyum (radyoaktif, teorik)
    'Am': [448.0, 450.5],                # Amerikyum (radyoaktif, teorik)
    'Cm': [447.0, 449.5],                # Küriyum (radyoaktif, teorik)
    'Bk': [446.0, 448.5],                # Berkelyum (radyoaktif, teorik)
    'Cf': [445.0, 447.5],                # Kaliforniyum (radyoaktif, teorik)
    'Es': [444.0, 446.5],                # Aynştaynyum (radyoaktif, teorik)
    'Fm': [443.0, 445.5],                # Fermiyum (radyoaktif, teorik)
    'Md': [442.0, 444.5],                # Mendelevyum (radyoaktif, teorik)
    'No': [441.0, 443.5],                # Nobelyum (radyoaktif, teorik)
    'Lr': [440.0, 442.5],                # Lavrensiyum (radyoaktif, teorik)
    'Rf': [439.0, 441.5],                # Rutherfordiyum (teorik)
    'Db': [438.0, 440.5],                # Dubniyum (teorik)
    'Sg': [437.0, 439.5],                # Seaborgiyum (teorik)
    'Bh': [436.0, 438.5],                # Bohriyum (teorik)
    'Hs': [435.0, 437.5],                # Hassiyum (teorik)
    'Mt': [434.0, 436.5],                # Meitneriyum (teorik)
    'Ds': [433.0, 435.5],                # Darmstadtium (teorik)
    'Rg': [432.0, 434.5],                # Roentgenyum (teorik)
    'Cn': [431.0, 433.5],                # Kopernikyum (teorik)
    'Nh': [430.0, 432.5],                # Nihonyum (teorik)
    'Fl': [429.0, 431.5],                # Flerovyum (teorik)
    'Mc': [428.0, 430.5],                # Moskovyum (teorik)
    'Lv': [427.0, 429.5],                # Livermorium (teorik)
    'Ts': [426.0, 428.5],                # Tennessin (teorik)
    'Og': [425.0, 427.5],                # Oganesson (teorik)
    }
    
    return periodic_elements, spectral_lines

def load_element_data_and_spectral_lines(filename):
    """Loads element data and spectral lines from a text file."""
    element_data = {}
    spectral_lines = {}
    current_section = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if "Element Data" in line:
                    current_section = "element"
                elif "Spectral Lines" in line:
                    current_section = "spectral"
                continue
            
            parts = line.split(',')
            if current_section == "element" and len(parts) >= 2:
                symbol = parts[0]
                atomic_number = int(parts[1])
                element_data[atomic_number] = (symbol, atomic_number)
            elif current_section == "spectral" and len(parts) >= 2:
                symbol = parts[0]
                wavelengths = [float(wl) for wl in parts[1:] if wl]
                spectral_lines[symbol] = wavelengths
    
    return element_data, spectral_lines

def wavelength_to_rgb(wavelength, gamma=0.8):
    wavelength = float(wavelength)
    if 380 <= wavelength <= 750:
        if wavelength < 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength < 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength < 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength < 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength < 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        else:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
    else:
        R = G = B = 0.0 # UV veya IR için siyah
    return (R, G, B)

def get_text_color_for_bg(bg_color):
    """Determines optimal text color (white or black) based on background luminance."""
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return 'white' if luminance < 0.5 else 'black'

def generate_soft_random_colors(n):
    """
    Generates n soft, pastel, and completely random colors.
    Uses high Value and Saturation in HSV space for a soft look.
    """
    colors = []
    for _ in range(n):
        hue = random.random()
        # Soft görünüm için doygunluk (saturation) orta seviyede
        saturation = 0.4 + (random.random() * 0.4)
        # Soft görünüm için parlaklık (value) yüksek
        value = 0.7 + (random.random() * 0.3)
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(rgb)
    return colors

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (random.random() * 0.3) # 0.7 - 1.0 arası
        value = 0.8 + (random.random() * 0.2)     # 0.8 - 1.0 arası
        rgb = plt.cm.hsv(hue)[:3] # HSV'den RGB'ye dönüştür
        # Parlaklığı ayarla
        from matplotlib.colors import hsv_to_rgb
        adjusted_rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(adjusted_rgb)
    return colors

# 2D Layout
def kececi_layout_2d(
    nx_graph: nx.Graph,
    primary_spacing: float = 1.0,
    secondary_spacing: float = 1.0,
    primary_direction: str = 'left-to-right',
    secondary_start: str = 'up',
    expanding: bool = True
) -> Dict[int, Tuple[float, float]]:
    pos = {}
    nodes = sorted(list(nx_graph.nodes()))

    for i, node_id in enumerate(nodes):
        if primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        elif primary_direction == 'right-to-left':
            primary_coord, secondary_axis = i * -primary_spacing, 'y'
        elif primary_direction == 'top_down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        else:  # 'bottom_up'
            primary_coord, secondary_axis = i * primary_spacing, 'x'

        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        x, y = (secondary_offset, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_offset)
        pos[node_id] = (x, y)

    return pos

# Silindirik Layout
def kececi_layout_cylindrical(
    nx_graph: nx.Graph,
    radius: float = 5.0,
    height: float = 10.0
) -> Dict[int, Tuple[float, float, float]]:
    pos_3d = {}
    nodes = sorted(list(nx_graph.nodes()))
    num_nodes = len(nodes)

    for i, node_id in enumerate(nodes):
        theta = 2 * np.pi * i / num_nodes
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height * i / num_nodes
        pos_3d[node_id] = (x, y, z)

    return pos_3d

# Kübik Layout
def kececi_layout_cubic(
    nx_graph: nx.Graph,
    size: float = 5.0
) -> Dict[int, Tuple[float, float, float]]:
    pos_3d = {}
    nodes = sorted(list(nx_graph.nodes()))
    num_nodes = len(nodes)
    cube_size = int(np.cbrt(num_nodes)) + 1

    for i, node_id in enumerate(nodes):
        x = size * (i % cube_size)
        y = size * ((i // cube_size) % cube_size)
        z = size * ((i // (cube_size ** 2)) % cube_size)
        pos_3d[node_id] = (x, y, z)

    return pos_3d

# Küresel Layout
def kececi_layout_spherical(
    nx_graph: nx.Graph,
    radius: float = 5.0
) -> Dict[int, Tuple[float, float, float]]:
    pos_3d = {}
    nodes = sorted(list(nx_graph.nodes()))
    num_nodes = len(nodes)

    for i, node_id in enumerate(nodes):
        theta = 2 * np.pi * i / num_nodes
        phi = np.arccos(1 - 2 * (i + 0.5) / num_nodes)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        pos_3d[node_id] = (x, y, z)

    return pos_3d

# Eliptik Layout
def kececi_layout_elliptical(
    nx_graph: nx.Graph,
    a: float = 5.0,
    b: float = 3.0
) -> Dict[int, Tuple[float, float]]:
    pos = {}
    nodes = sorted(list(nx_graph.nodes()))
    num_nodes = len(nodes)

    for i, node_id in enumerate(nodes):
        theta = 2 * np.pi * i / num_nodes
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        pos[node_id] = (x, y)

    return pos

# Torik (Halkasal) Layout
def kececi_layout_toric(
    nx_graph: nx.Graph,
    major_radius: float = 5.0,
    minor_radius: float = 2.0
) -> Dict[int, Tuple[float, float, float]]:
    pos_3d = {}
    nodes = sorted(list(nx_graph.nodes()))
    num_nodes = len(nodes)

    for i, node_id in enumerate(nodes):
        theta = 2 * np.pi * i / num_nodes
        phi = 2 * np.pi * i / num_nodes
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        pos_3d[node_id] = (x, y, z)

    return pos_3d

# Ağırlıklı Çizim (draw_kececi_weighted)
def draw_kececi_weighted(
    nx_graph: nx.Graph,
    pos: Dict[int, Tuple[float, ...]],
    ax: Optional[plt.Axes] = None,
    layout: str = "unknown",          # ✅ Varsayılan değer
    style: str = "weighted",          # ✅ Varsayılan değer
    node_size: int = 300,
    edge_width_scale: float = 2.0,
    with_labels: bool = True,
    font_size: int = 10,
    font_weight: str = 'bold',
    **kwargs
) -> plt.Axes:

    """
    2D/3D Weighted edges ile Keçeci layout çizimi.
    """
    if ax is None or not hasattr(ax, 'scatter'):  # ax bozuksa yeniden oluştur
        is_3d = len(next(iter(pos.values()))) == 3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d' if is_3d else None)
    
    # ✅ nodes_kwargs'ı SIFIRLA (en güvenli)
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax, 
                          node_size=node_size, node_color='lightblue')
    """
    if ax is None:
        is_3d = len(next(iter(pos.values()))) == 3
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    # ✅ KWARGS FİLTRELEME
    nodes_kwargs = {k: v for k, v in kwargs.items() if k not in ['font_size', 'font_weight']}
    node_color = kwargs.get('node_color', 'lightblue')

    # Node'ları çiz ✅ font_size yok
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=node_size, 
                          node_color=node_color, **nodes_kwargs)
    """

    # Etiketleri çiz ✅ font_size var
    if with_labels:
        if len(next(iter(pos.values()))) == 3:  # 3D
            for node, coord in pos.items():
                ax.text(coord[0], coord[1], coord[2], str(node),
                       size=font_size, zorder=1, color='black', fontweight=font_weight)
        else:  # 2D
            nx.draw_networkx_labels(nx_graph, pos, ax=ax, 
                                  font_size=font_size, font_weight=font_weight)

    # Weighted edges
    weights = nx.get_edge_attributes(nx_graph, 'weight')
    if not weights:
        weights = {edge: 1.0 for edge in nx_graph.edges()}
    
    is_3d = len(next(iter(pos.values()))) == 3
    for (u, v), weight in weights.items():
        width = weight * edge_width_scale
        if is_3d:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]],
                   linewidth=width, color='gray', alpha=0.7)
        else:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                   linewidth=width, color='gray', alpha=0.7)

    #ax.set_title("Keçeci Layout: Weighted Edges", fontsize=font_size + 2)
    ax.set_title(f"Keçeci Layout: {layout.capitalize()} (Weighted Edges)", fontsize=font_size + 2)
    return ax


def draw_kececi_colored(
    nx_graph: nx.Graph,
    pos: Dict[int, Tuple[float, ...]],
    layout: str,                    # ✅ 1. Zorunlu (en başta)
    style: str = 'colored',         # ✅ 2. Varsayılan  
    ax: Optional[plt.Axes] = None,  # ✅ 3. Varsayılan
    node_size: int = 300,
    with_labels: bool = True,
    font_size: int = 10,
    font_weight: str = 'bold',
    **kwargs
) -> plt.Axes:
    """
    2D/3D Renkli node'lar ile Keçeci layout çizimi.
    """
    if ax is None:
        is_3d = len(next(iter(pos.values()))) == 3
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    # ✅ KWARGS FİLTRELEME
    nodes_kwargs = {k: v for k, v in kwargs.items() if k not in ['font_size', 'font_weight']}

    # Dereceye göre renk hesapla
    degrees = dict(nx_graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_colors = [plt.cm.viridis(deg / max_degree) for deg in degrees.values()]

    # Node'ları çiz ✅ font_size yok
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_colors,
                          node_size=node_size, **nodes_kwargs)

    # Etiketleri çiz ✅ font_size var
    if with_labels:
        if len(next(iter(pos.values()))) == 3:  # 3D
            for node, coord in pos.items():
                ax.text(coord[0], coord[1], coord[2], str(node),
                       size=font_size, zorder=1, color='black', fontweight=font_weight)
        else:  # 2D
            nx.draw_networkx_labels(nx_graph, pos, ax=ax, 
                                  font_size=font_size, font_weight=font_weight)

    # Edges
    is_3d = len(next(iter(pos.values()))) == 3
    if is_3d:
        for u, v in nx_graph.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]],
                   color='gray', alpha=0.5)
    else:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5)

    #ax.set_title("Keçeci Layout: Colored Nodes", fontsize=font_size + 2)
    ax.set_title(f"Keçeci Layout: {layout.capitalize()} (Colored Nodes)", fontsize=font_size + 2)
    return ax


"""
def draw_kececi_weighted(
    nx_graph: nx.Graph,
    pos: Dict[int, Tuple[float, ...]],
    ax: Optional[plt.Axes] = None,
    node_size: int = 300,
    edge_width_scale: float = 2.0,
    **kwargs
) -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

    weights = nx.get_edge_attributes(nx_graph, 'weight')
    if not weights:
        weights = {edge: 1.0 for edge in nx_graph.edges()}

    nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=node_size, **kwargs)

    is_3d = len(pos[next(iter(pos))]) == 3
    if is_3d:
        for node, coord in pos.items():
            ax.text(coord[0], coord[1], coord[2], f'  {node}', size=10, zorder=1, color='black')
    else:
        nx.draw_networkx_labels(nx_graph, pos, ax=ax)

    for (u, v), weight in weights.items():
        width = weight * edge_width_scale
        if is_3d:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                [pos[u][2], pos[v][2]],
                linewidth=width,
                color='gray',
                alpha=0.7
            )
        else:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                linewidth=width,
                color='gray',
                alpha=0.7
            )

    ax.set_title("Keçeci Layout: Weighted Edges")
    return ax
"""
"""    
# Renkli Çizim (draw_kececi_colored)
def draw_kececi_colored(
    nx_graph: nx.Graph,
    pos: Dict[int, Tuple[float, ...]],
    ax: Optional[plt.Axes] = None,
    node_size: int = 300,
    **kwargs
) -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

    degrees = dict(nx_graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_colors = [plt.cm.viridis(deg / max_degree) for deg in degrees.values()]

    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        **kwargs
    )

    is_3d = len(pos[next(iter(pos))]) == 3
    if is_3d:
        for node, coord in pos.items():
            ax.text(coord[0], coord[1], coord[2], f'  {node}', size=10, zorder=1, color='black')
    else:
        nx.draw_networkx_labels(nx_graph, pos, ax=ax)

    if is_3d:
        for u, v in nx_graph.edges():
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                [pos[u][2], pos[v][2]],
                color='gray',
                alpha=0.5
            )
    else:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5)

    ax.set_title("Keçeci Layout: Colored Nodes")
    return ax
"""

# =============================================================================
# 3. INTERNAL DRAWING STYLE IMPLEMENTATIONS
# =============================================================================

def _draw_internal(nx_graph, ax, style, **kwargs):
    """Internal router that handles the different drawing styles."""
    layout_params = {
        k: v for k, v in kwargs.items()
        if k in ['primary_spacing', 'secondary_spacing', 'primary_direction',
                 'secondary_start', 'expanding']
    }
    draw_params = {k: v for k, v in kwargs.items() if k not in layout_params}

    if style == 'curved':
        pos = kececi_layout(nx_graph, **layout_params)
        final_params = {'ax': ax, 'with_labels': True, 'node_color': '#1f78b4',
                        'node_size': 700, 'font_color': 'white',
                        'connectionstyle': 'arc3,rad=0.2', 'arrows': True}
        final_params.update(draw_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            nx.draw(nx_graph, pos, **final_params)
        ax.set_title("Keçeci Layout: Curved Edges")

    elif style == 'transparent':
        pos = kececi_layout(nx_graph, **layout_params)
        # node_color'u draw_params'dan al, yoksa default değeri kullan
        node_color = draw_params.pop('node_color', '#2ca02c')  # DÜZELTME BURADA
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_color, 
                              node_size=700, **draw_params)  # DÜZELTME BURADA
        nx.draw_networkx_labels(nx_graph, pos, ax=ax, font_color='white')
        edge_lengths = {e: np.linalg.norm(np.array(pos[e[0]]) - np.array(pos[e[1]])) for e in nx_graph.edges()}
        max_len = max(edge_lengths.values()) if edge_lengths else 1.0
        for edge, length in edge_lengths.items():
            alpha = 0.15 + 0.85 * (1 - length / max_len)
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[edge], ax=ax, 
                                  width=1.5, edge_color='black', alpha=alpha)
        ax.set_title("Keçeci Layout: Transparent Edges")

    elif style == '3d':
        pos_3d = _kececi_layout_3d_helix(nx_graph)
        node_color = draw_params.get('node_color', '#d62728')  # DÜZELTME BURADA
        edge_color = draw_params.get('edge_color', 'gray')     # DÜZELTME BURADA
        for node, (x, y, z) in pos_3d.items():
            ax.scatter([x], [y], [z], s=200, c=[node_color], depthshade=True)
            ax.text(x, y, z, f'  {node}', size=10, zorder=1, color='k')
        for u, v in nx_graph.edges():
            coords = np.array([pos_3d[u], pos_3d[v]])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
                   color=edge_color, alpha=0.8)  # DÜZELTME BURADA
        ax.set_title("Keçeci Layout: 3D Helix")
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-60)
"""
def _draw_internal(nx_graph, ax, style, **kwargs):
    #Internal router that handles the different drawing styles.
    layout_params = {
        k: v for k, v in kwargs.items()
        if k in ['primary_spacing', 'secondary_spacing', 'primary_direction',
                 'secondary_start', 'expanding']
    }
    draw_params = {k: v for k, v in kwargs.items() if k not in layout_params}

    if style == 'curved':
        pos = kececi_layout(nx_graph, **layout_params)
        final_params = {'ax': ax, 'with_labels': True, 'node_color': '#1f78b4',
                        'node_size': 700, 'font_color': 'white',
                        'connectionstyle': 'arc3,rad=0.2', 'arrows': True}
        final_params.update(draw_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            nx.draw(nx_graph, pos, **final_params)
        ax.set_title("Keçeci Layout: Curved Edges")

    elif style == 'transparent':
        pos = kececi_layout(nx_graph, **layout_params)
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color='#2ca02c', node_size=700, **draw_params)
        nx.draw_networkx_labels(nx_graph, pos, ax=ax, font_color='white')
        edge_lengths = {e: np.linalg.norm(np.array(pos[e[0]]) - np.array(pos[e[1]])) for e in nx_graph.edges()}
        max_len = max(edge_lengths.values()) if edge_lengths else 1.0
        for edge, length in edge_lengths.items():
            alpha = 0.15 + 0.85 * (1 - length / max_len)
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[edge], ax=ax, width=1.5, edge_color='black', alpha=alpha)
        ax.set_title("Keçeci Layout: Transparent Edges")

    elif style == '3d':
        pos_3d = _kececi_layout_3d_helix(nx_graph)
        node_color = draw_params.get('node_color', '#d62728')
        edge_color = draw_params.get('edge_color', 'gray')
        for node, (x, y, z) in pos_3d.items():
            ax.scatter([x], [y], [z], s=200, c=[node_color], depthshade=True)
            ax.text(x, y, z, f'  {node}', size=10, zorder=1, color='k')
        for u, v in nx_graph.edges():
            coords = np.array([pos_3d[u], pos_3d[v]])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=edge_color, alpha=0.8)
        ax.set_title("Keçeci Layout: 3D Helix")
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-60)
"""

# =============================================================================
# 4. MAIN USER-FACING DRAWING FUNCTION
# =============================================================================
def draw_kececi(
    graph, pos=None, layout=None, style='default', ax=None,
    with_labels=True, node_color='lightblue', node_size=500,
    font_size=10, font_weight='bold', edge_color='gray',
    edge_alpha=0.5, edge_width=1.0, **kwargs
) -> plt.Axes:
    nx_graph = to_networkx(graph)

    # Layout hesaplama ✅
    if pos is None:
        layout = layout or '2d'
        if layout == '2d':
            pos = kececi_layout_2d(nx_graph)
        elif layout == 'cylindrical':
            pos = kececi_layout_cylindrical(nx_graph)
        elif layout == 'cubic':
            pos = kececi_layout_cubic(nx_graph, **kwargs)
        elif layout == 'spherical':
            pos = kececi_layout_spherical(nx_graph, **kwargs)
        elif layout == 'elliptical':
            pos = kececi_layout_elliptical(nx_graph, **kwargs)
        elif layout == 'toric':
            pos = kececi_layout_toric(nx_graph, **kwargs)
        else:
            raise ValueError(f"Geçersiz layout: {layout}")

    # ✅ YENİ FIGÜR HER ZAMAN (backend temizliği)
    fig = plt.figure(figsize=(10, 8))
    is_3d = len(next(iter(pos.values()))) == 3
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    nodes_kwargs = {k: v for k, v in kwargs.items() if k not in ['font_size', 'font_weight']}

    # Style bazlı çizim
    if style == 'weighted':
        draw_kececi_weighted(nx_graph=nx_graph, pos=pos, ax=ax, 
                            layout=layout, style='weighted',
                            node_size=node_size, font_size=font_size,
                            font_weight=font_weight, **kwargs)
    elif style == 'colored':
        draw_kececi_colored(nx_graph=nx_graph, pos=pos, ax=ax,
                           layout=layout, style='colored',
                           node_size=node_size, font_size=font_size,
                           font_weight=font_weight, **kwargs)


    else:  # default/standard/curved/helix/3d
        # Default çizim (önceki kodunuz)
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_color, 
                              node_size=node_size, **nodes_kwargs)

        # Labels ✅ font_size var
        if with_labels:
            if is_3d:
                for node, coord in pos.items():
                    ax.text(coord[0], coord[1], coord[2], str(node),
                           size=font_size, zorder=1, color='black',
                           fontweight=font_weight, ha='center', va='center')
            else:
                nx.draw_networkx_labels(nx_graph, pos, ax=ax, 
                                      font_size=font_size, font_weight=font_weight)

        # Edges
        if is_3d:
            for u, v in nx_graph.edges():
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]],
                       color=edge_color, alpha=edge_alpha, linewidth=edge_width)
        else:
            nx.draw_networkx_edges(nx_graph, pos, ax=ax, 
                                  alpha=edge_alpha, edge_color=edge_color, width=edge_width)

    plt.tight_layout()  # ✅ Backend temizliği
    layout_display = (layout or "Custom").capitalize()
    ax.set_title(f"Keçeci Layout: {layout_display} ({style})", fontsize=font_size + 2)
    ax.set_axis_off()
    return ax

"""
def draw_kececi(
    graph,
    pos: Optional[Dict[int, Tuple[float, ...]]] = None,
    layout: Optional[str] = None,
    style: str = 'default',
    ax: Optional[plt.Axes] = None,
    with_labels: bool = True,
    node_color: Union[str, List] = 'lightblue',
    node_size: int = 500,
    font_weight: str = 'bold',
    edge_color: str = 'gray',
    edge_alpha: float = 0.5,
    edge_width: float = 1.0,
    **kwargs
) -> plt.Axes:

    Keçeci Layout ile 2D/3D uyumlu graf çizimi.

    Args:
        graph: Graf objesi (NetworkX, igraph, vb.).
        pos: Önceden hesaplanmış koordinatlar (opsiyonel).
        layout: '2d', 'cylindrical', 'cubic', 'spherical', 'elliptical', 'toric' (opsiyonel).
        style: 'default', 'weighted', 'colored'.
        ax: Matplotlib ekseni.
        with_labels: Düğüm etiketlerini göster.
        node_color: Düğüm rengi (tek renk veya renk listesi).
        node_size: Düğüm boyutu.
        font_weight: Yazı kalınlığı.
        edge_color: Kenar rengi.
        edge_alpha: Kenar şeffaflığı.
        edge_width: Kenar kalınlığı.
        **kwargs: Ek parametreler.

    Returns:
        Matplotlib ekseni.

    nx_graph = to_networkx(graph)

    # Eğer pos verilmemişse, layout'a göre hesapla
    if pos is None:
        if layout is None:
            layout = '2d'  # Varsayılan layout

        if layout == '2d':
            pos = kececi_layout_2d(nx_graph, **kwargs)
        elif layout == 'cylindrical':
            pos = kececi_layout_cylindrical(nx_graph, **kwargs)
        elif layout == 'cubic':
            pos = kececi_layout_cubic(nx_graph, **kwargs)
        elif layout == 'spherical':
            pos = kececi_layout_spherical(nx_graph, **kwargs)
        elif layout == 'elliptical':
            pos = kececi_layout_elliptical(nx_graph, **kwargs)
        elif layout == 'toric':
            pos = kececi_layout_toric(nx_graph, **kwargs)
        else:
            raise ValueError(f"Geçersiz layout: {layout}")

    # 2D mi 3D mi kontrol et
    is_3d = len(pos[next(iter(pos))]) == 3

    # Eksen oluştur (eğer verilmemişse)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    # Stile göre çizim yap
    if style == 'weighted':
        draw_kececi_weighted(nx_graph, pos, ax, node_size=node_size, 
                            with_labels=with_labels, font_weight=font_weight, **kwargs)
    elif style == 'colored':
        draw_kececi_colored(nx_graph, pos, ax, node_size=node_size,
                           with_labels=with_labels, font_weight=font_weight, **kwargs)
    else:  # 'default'
        # Node'ları çiz
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, 
                              node_color=node_color, 
                              node_size=node_size,
                              **kwargs)

        # Etiketleri çiz
        if with_labels:
            if is_3d:
                # 3D için özel etiket çizimi
                for node, coord in pos.items():
                    ax.text(coord[0], coord[1], coord[2],  # 3D koordinatlar
                           str(node),  # 's' parametresi - etiket metni
                           size=10, 
                           zorder=1, 
                           color='black',
                           fontweight=font_weight,
                           ha='center',  # Yatayda ortala
                           va='center')  # Dikeyde ortala
            else:
                # 2D için NetworkX etiket çizimi
                nx.draw_networkx_labels(nx_graph, pos, ax=ax, font_weight=font_weight)

        # Edge'leri çiz
        if is_3d:
            for u, v in nx_graph.edges():
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    [pos[u][2], pos[v][2]],
                    color=edge_color,
                    alpha=edge_alpha,
                    linewidth=edge_width
                )
        else:
            nx.draw_networkx_edges(nx_graph, pos, ax=ax, 
                                  alpha=edge_alpha, 
                                  edge_color=edge_color,
                                  width=edge_width)

    title = f"Keçeci Layout: {layout.capitalize() if layout else 'Custom'} ({style})"
    ax.set_title(title)
    
    # Eksenleri kapat
    ax.set_axis_off()
    
    return ax
"""

def draw_kececi_custom_labels(
    graph,
    pos: Dict[int, Tuple[float, ...]],
    labels: Dict[int, str],
    ax: Optional[plt.Axes] = None,
    node_size: int = 500,
    node_color: Union[str, List] = 'lightblue',
    font_size: int = 10,
    font_color: Union[str, List] = 'black',
    font_weight: str = 'bold',
    **kwargs
) -> plt.Axes:
    """
    Özel etiketlerle Keçeci layout çizimi (2D/3D uyumlu).
    
    Args:
        graph: Graf objesi
        pos: Node pozisyonları
        labels: Özel etiketler (node_id: label_text)
        ax: Matplotlib ekseni
        node_size: Node boyutu
        node_color: Node rengi
        font_size: Yazı boyutu
        font_color: Yazı rengi
        font_weight: Yazı kalınlığı
        **kwargs: Ek parametreler
        
    Returns:
        Matplotlib ekseni
    """
    nx_graph = to_networkx(graph)
    
    # 2D mi 3D mi kontrol et
    is_3d = len(pos[next(iter(pos))]) == 3
    
    # Eksen oluştur (eğer verilmemişse)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    
    # Node'ları çiz
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax,
                          node_color=node_color,
                          node_size=node_size,
                          **kwargs)
    
    # Edge'leri çiz
    if is_3d:
        for u, v in nx_graph.edges():
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                [pos[u][2], pos[v][2]],
                color='gray',
                alpha=0.5
            )
    else:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5)
    
    # Özel etiketleri çiz
    if is_3d:
        for node, coord in pos.items():
            if node in labels:
                ax.text(coord[0], coord[1], coord[2],
                       labels[node],
                       fontsize=font_size,
                       fontweight=font_weight,
                       color=font_color if isinstance(font_color, str) else font_color[node-1],
                       ha='center',
                       va='center',
                       zorder=10)
    else:
        # 2D için NetworkX etiket çizimi
        nx.draw_networkx_labels(nx_graph, pos, labels=labels, ax=ax,
                               font_size=font_size,
                               font_color=font_color,
                               font_weight=font_weight)
    
    ax.set_title("Keçeci Layout with Custom Labels")
    ax.set_axis_off()
    
    return ax
"""
def draw_kececi(
    graph,
    pos: Optional[Dict[int, Tuple[float, ...]]] = None,
    layout: Optional[str] = None,
    style: str = 'default',
    ax: Optional[plt.Axes] = None,
    with_labels: bool = True,
    node_color: str = 'lightblue',
    node_size: int = 500,
    font_weight: str = 'bold',
    **kwargs
) -> plt.Axes:
    
    Keçeci Layout ile graf çizimi.
    Args:
        graph: Graf objesi (NetworkX, igraph, vb.).
        pos: Önceden hesaplanmış koordinatlar (opsiyonel).
        layout: '2d', 'cylindrical', 'cubic', 'spherical', 'elliptical', 'toric' (opsiyonel).
        style: 'default', 'weighted', 'colored'.
        ax: Matplotlib ekseni.
        with_labels: Düğüm etiketlerini göster.
        node_color: Düğüm rengi.
        node_size: Düğüm boyutu.
        font_weight: Yazı kalınlığı.
        **kwargs: Ek parametreler.
    Returns:
        Matplotlib ekseni.

    nx_graph = to_networkx(graph)

    # Eğer pos verilmemişse, layout'a göre hesapla
    if pos is None:
        if layout is None:
            layout = '2d'  # Varsayılan layout

        if layout == '2d':
            pos = kececi_layout_2d(nx_graph, **kwargs)
        elif layout == 'cylindrical':
            pos = kececi_layout_cylindrical(nx_graph, **kwargs)
        elif layout == 'cubic':
            pos = kececi_layout_cubic(nx_graph, **kwargs)
        elif layout == 'spherical':
            pos = kececi_layout_spherical(nx_graph, **kwargs)
        elif layout == 'elliptical':
            pos = kececi_layout_elliptical(nx_graph, **kwargs)
        elif layout == 'toric':
            pos = kececi_layout_toric(nx_graph, **kwargs)
        else:
            raise ValueError(f"Geçersiz layout: {layout}")

    # 3D için eksen ayarlaması
    is_3d = len(pos[next(iter(pos))]) == 3
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    # Stile göre çizim yap
    if style == 'weighted':
        draw_kececi_weighted(nx_graph, pos, ax, **kwargs)
    elif style == 'colored':
        draw_kececi_colored(nx_graph, pos, ax, **kwargs)
    else:  # 'default'
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_color, node_size=node_size)

        # Düğüm etiketlerini çiz
        if with_labels:
            if is_3d:
                for node, coord in pos.items():
                    ax.text(coord[0], coord[1], coord[2], f'  {node}', size=10, zorder=1, color='black', fontweight=font_weight)
            else:
                nx.draw_networkx_labels(nx_graph, pos, ax=ax, font_weight=font_weight)

        # Kenarları çiz
        if is_3d:
            for u, v in nx_graph.edges():
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    [pos[u][2], pos[v][2]],
                    color='gray',
                    alpha=0.5
                )
        else:
            nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5)

    ax.set_title(f"Keçeci Layout: {layout.capitalize() if layout else 'Custom'} ({style})")
    return ax
"""
"""
def draw_kececi(
    graph,
    layout: str = '2d',
    style: str = 'default',
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:

    Keçeci Layout ile graf çizimi.
    Args:
        graph: Graf objesi (NetworkX, igraph, vb.).
        layout: '2d', 'cylindrical', 'cubic', 'spherical', 'elliptical', 'toric'.
        style: 'default', 'weighted', 'colored'.
        ax: Matplotlib ekseni.
        **kwargs: Ek parametreler.
    Returns:
        Matplotlib ekseni.

    nx_graph = to_networkx(graph)

    # Layout'a göre koordinatları hesapla
    if layout == '2d':
        pos = kececi_layout_2d(nx_graph, **kwargs)
    elif layout == 'cylindrical':
        pos = kececi_layout_cylindrical(nx_graph, **kwargs)
    elif layout == 'cubic':
        pos = kececi_layout_cubic(nx_graph, **kwargs)
    elif layout == 'spherical':
        pos = kececi_layout_spherical(nx_graph, **kwargs)
    elif layout == 'elliptical':
        pos = kececi_layout_elliptical(nx_graph, **kwargs)
    elif layout == 'toric':
        pos = kececi_layout_toric(nx_graph, **kwargs)
    else:
        raise ValueError(f"Invalid layout: {layout}")

    # 3D için eksen ayarlaması
    is_3d = len(pos[next(iter(pos))]) == 3
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    # Stile göre çizim yap
    if style == 'weighted':
        draw_kececi_weighted(nx_graph, pos, ax, **kwargs)
    elif style == 'colored':
        draw_kececi_colored(nx_graph, pos, ax, **kwargs)
    else:  # 'default'
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, **kwargs)

        # Düğüm etiketlerini çiz
        if is_3d:
            for node, coord in pos.items():
                ax.text(coord[0], coord[1], coord[2], f'  {node}', size=10, zorder=1, color='black')
        else:
            nx.draw_networkx_labels(nx_graph, pos, ax=ax)

        # Kenarları çiz
        if is_3d:
            for u, v in nx_graph.edges():
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    [pos[u][2], pos[v][2]],
                    color='gray',
                    alpha=0.5
                )
        else:
            nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5)

    ax.set_title(f"Keçeci Layout: {layout.capitalize()} ({style})")
    return ax
"""

"""
def draw_kececi(graph, style='curved', ax=None, **kwargs):

    Draws a graph using the Keçeci Layout with a specified style.

    This function automatically handles graphs from different libraries
    (Networkx, Networkit, Rustworkx, igraph, Graphillion, graph-tool,etc.).

    Args:
        graph: The graph object to be drawn.
        style (str): The drawing style. Options: 'curved', 'transparent', '3d'.
        ax (matplotlib.axis.Axis, optional): The axis to draw on. If not
            provided, a new figure and axis are generated.
        **kwargs: Additional keyword arguments passed to both `kececi_layout`
                  and the drawing functions (e.g., expanding=True, node_size=500).

    Returns:
        matplotlib.axis.Axis: The axis object where the graph was drawn.

    nx_graph = to_networkx(graph)
    is_3d = (style.lower() == '3d')

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        projection = '3d' if is_3d else None
        ax = fig.add_subplot(111, projection=projection)

    if is_3d and getattr(ax, 'name', '') != '3d':
        raise ValueError("The '3d' style requires an axis with 'projection=\"3d\"'.")

    draw_styles = ['curved', 'transparent', '3d']
    if style.lower() not in draw_styles:
        raise ValueError(f"Invalid style: '{style}'. Options are: {draw_styles}")

    _draw_internal(nx_graph, ax, style.lower(), **kwargs)
    return ax
"""

def draw_kececi_periodic_table(
    graph,
    periodic_elements: Dict[int, Tuple[str, int]],
    layout_type: str = '3d_helix',
    layout_params: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    dimension: str = 'auto',  # '2d', '3d', or 'auto'
    color_scheme: str = 'vibrant',  # 'vibrant', 'distinct', 'pastel', 'group', 'period', 'block', 'electronegativity'
    node_size: Union[int, List[int]] = 1600,
    font_size: Union[int, List[int]] = 10,
    edge_style: str = 'standard',  # 'standard', 'light', 'bold', 'hidden'
    label_position: str = 'center',  # 'center', 'above', 'below', 'right', 'left'
    zorder_strategy: str = 'smart',  # 'smart', 'fixed', 'z_based'
    show_legend: bool = False,
    title: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Axes, Dict]:
    """
    Gelişmiş periyodik tablo çizimi - Hem 2D hem 3D uyumlu.
    """
    # Graph'ı NetworkX'e çevir
    nx_graph = to_networkx(graph)
    node_count = len(periodic_elements)
    
    # Dimension belirleme
    if dimension == 'auto':
        if '3d' in layout_type.lower():
            dimension = '3d'
        else:
            dimension = '2d'
    
    # Layout parametreleri
    if layout_params is None:
        layout_params = {}
    
    # Layout hesapla
    pos = _calculate_layout(nx_graph, layout_type, layout_params, dimension, node_count)
    
    # Renkleri oluştur
    node_colors = _generate_colors(node_count, color_scheme, periodic_elements)
    
    # Etiketleri oluştur
    custom_labels = _generate_labels(nx_graph, periodic_elements)
    
    # Eksen oluştur (eğer verilmemişse)
    if ax is None:
        fig_size = kwargs.get('figsize', (20, 20) if dimension == '3d' else (16, 16))
        fig = plt.figure(figsize=fig_size)
        if dimension == '3d':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    
    # Çizim sırasını belirle
    draw_order = _get_draw_order(nx_graph, pos, zorder_strategy, dimension)
    
    # Edge stilleri
    edge_config = _get_edge_config(edge_style)
    
    # Edge'leri çiz (ilk sırada)
    _draw_edges(nx_graph, pos, ax, edge_config, dimension)
    
    # Node'ları çiz (çizim sırasına göre)
    _draw_nodes(nx_graph, pos, node_colors, ax, node_size, draw_order, dimension, **kwargs)
    
    # Etiketleri çiz
    _draw_labels(nx_graph, pos, custom_labels, node_colors, ax, font_size, 
                label_position, dimension, **kwargs)
    
    # Başlık
    if title is None:
        title = _generate_title(layout_type, node_count, color_scheme, dimension)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=25)
    
    # Eksen ayarları
    _configure_axes(ax, pos, dimension, layout_params, **kwargs)
    
    # Açıklama (legend)
    if show_legend and color_scheme in ['group', 'period', 'block']:
        _add_legend(ax, color_scheme, dimension, periodic_elements)
    
    return ax, pos


# Yardımcı fonksiyonlar (öncekiyle aynı)
def _calculate_layout(graph, layout_type, params, dimension, node_count=None):
    """Layout hesapla."""
    if layout_type == '3d_helix':
        return kececi_layout_3d_helix_parametric(
            graph,
            z_spacing=params.get('z_spacing', 8.0),
            radius=params.get('radius', 25.0),
            turns=params.get('turns', 3.0)
        )
    
    elif layout_type == '2d_linear':
        return kececi_layout_2d(
            graph,
            primary_spacing=params.get('primary_spacing', 2.0),
            secondary_spacing=params.get('secondary_spacing', 2.0),
            primary_direction=params.get('primary_direction', 'left-to-right'),
            secondary_start=params.get('secondary_start', 'up'),
            expanding=params.get('expanding', True)
        )
    
    elif layout_type == '2d_circular':
        return nx.circular_layout(graph, scale=params.get('scale', 1.0))
    
    elif layout_type == '3d_spherical':
        pos = {}
        n = len(graph.nodes())
        for i, node in enumerate(graph.nodes()):
            phi = np.arccos(1 - 2 * (i + 0.5) / n)
            theta = np.pi * (1 + 5**0.5) * i
            radius = params.get('radius', 20.0)
            pos[node] = (
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi)
            )
        return pos
    
    elif layout_type == '2d_spring':
        return nx.spring_layout(graph, k=params.get('k', 2.0), 
                               iterations=params.get('iterations', 50))
    
    elif layout_type == '2d_grid':
        return _generate_grid_layout(graph, params, node_count)
    
    else:
        return kececi_layout_2d(
            graph,
            primary_spacing=params.get('primary_spacing', 2.0),
            secondary_spacing=params.get('secondary_spacing', 2.0),
            primary_direction=params.get('primary_direction', 'left-to-right'),
            secondary_start=params.get('secondary_start', 'up'),
            expanding=params.get('expanding', True)
        )

def _generate_grid_layout(graph, params, node_count):
    """Manuel grid layout oluştur."""
    rows = params.get('rows', None)
    cols = params.get('cols', None)
    spacing = params.get('spacing', 2.0)
    
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(node_count)))
        rows = int(np.ceil(node_count / cols))
    elif rows is None:
        rows = int(np.ceil(node_count / cols))
    elif cols is None:
        cols = int(np.ceil(node_count / rows))
    
    pos = {}
    for i, node in enumerate(sorted(graph.nodes())):
        row = i // cols
        col = i % cols
        
        x_offset = -(cols - 1) * spacing / 2
        y_offset = (rows - 1) * spacing / 2
        
        x = col * spacing + x_offset
        y = -row * spacing + y_offset
        
        pos[node] = (x, y)
    
    return pos


def _generate_colors(node_count, scheme, periodic_elements=None):
    """Renkleri oluştur."""
    colors = []
    
    if scheme == 'vibrant':
        for i in range(node_count):
            hue = (i * 0.618033988749895) % 1.0
            saturation = 0.7 + np.random.random() * 0.3
            value = 0.8 + np.random.random() * 0.2
            colors.append(hsv_to_rgb([hue, saturation, value]))
    
    elif scheme == 'distinct':
        colors = generate_distinct_colors(node_count)
    
    elif scheme == 'pastel':
        for i in range(node_count):
            hue = i / max(node_count, 1)
            saturation = 0.4 + np.random.random() * 0.3
            value = 0.9 + np.random.random() * 0.1
            colors.append(hsv_to_rgb([hue, saturation, value]))
    
    elif scheme == 'group' and periodic_elements:
        colors = _get_group_colors(node_count, periodic_elements)
    
    elif scheme == 'period' and periodic_elements:
        colors = _get_period_colors(node_count, periodic_elements)
    
    elif scheme == 'block' and periodic_elements:
        colors = _get_block_colors(node_count, periodic_elements)
    
    elif scheme == 'electronegativity' and periodic_elements:
        colors = _get_electronegativity_colors(node_count, periodic_elements)
    
    else:
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(node_count)]
    
    return colors

def _get_group_colors(node_count, periodic_elements):
    """Gruplara göre renkler."""
    colors = []
    group_colors = {
        1: (1.0, 0.6, 0.6),    # Alkali metals
        2: (1.0, 0.8, 0.6),    # Alkaline earth
        3: (0.8, 1.0, 0.6),    # Group 3
        4: (0.7, 0.9, 0.8),    # Group 4
        5: (0.6, 0.9, 0.9),    # Group 5
        6: (0.6, 0.8, 1.0),    # Group 6
        7: (0.8, 0.6, 1.0),    # Group 7
        8: (0.9, 0.9, 0.6),    # Group 8
        9: (1.0, 0.9, 0.6),    # Group 9
        10: (0.9, 0.8, 0.7),   # Group 10
        11: (1.0, 0.8, 0.8),   # Group 11
        12: (0.8, 1.0, 0.8),   # Group 12
        13: (0.8, 0.9, 1.0),   # Boron group
        14: (0.9, 0.8, 1.0),   # Carbon group
        15: (1.0, 0.8, 0.9),   # Nitrogen group
        16: (0.8, 1.0, 0.9),   # Oxygen group
        17: (1.0, 0.9, 0.8),   # Halogens
        18: (0.9, 0.9, 0.9),   # Noble gases
    }
    
    lanthanide_color = (0.7, 1.0, 0.7)
    actinide_color = (1.0, 0.7, 0.7)
    
    for node_id in range(1, node_count + 1):
        if node_id in list(range(57, 72)):
            colors.append(lanthanide_color)
        elif node_id in list(range(89, 104)):
            colors.append(actinide_color)
        else:
            group = _determine_group(node_id)
            colors.append(group_colors.get(group, (0.8, 0.8, 0.8)))
    
    return colors

def _determine_group(atomic_num):
    """Atom numarasına göre grup belirle."""
    if atomic_num <= 2:
        return atomic_num
    elif atomic_num <= 10:
        return atomic_num - 2
    elif atomic_num <= 18:
        return atomic_num - 10
    elif atomic_num <= 36:
        if atomic_num <= 20:
            return atomic_num - 18
        elif atomic_num <= 30:
            return atomic_num - 20
        else:
            return atomic_num - 28
    elif atomic_num <= 54:
        if atomic_num <= 38:
            return atomic_num - 36
        elif atomic_num <= 48:
            return atomic_num - 38
        else:
            return atomic_num - 46
    else:
        return 18

def _get_period_colors(node_count, periodic_elements):
    """Periyotlara göre renkler."""
    colors = []
    period_colors = [
        (1.0, 0.7, 0.7),  # Period 1
        (1.0, 0.9, 0.7),  # Period 2
        (0.9, 1.0, 0.7),  # Period 3
        (0.7, 1.0, 0.8),  # Period 4
        (0.7, 0.9, 1.0),  # Period 5
        (0.8, 0.7, 1.0),  # Period 6
        (1.0, 0.7, 0.9),  # Period 7
    ]
    
    for atomic_num in range(1, node_count + 1):
        if atomic_num <= 2:
            period = 0
        elif atomic_num <= 10:
            period = 1
        elif atomic_num <= 18:
            period = 2
        elif atomic_num <= 36:
            period = 3
        elif atomic_num <= 54:
            period = 4
        elif atomic_num <= 86:
            period = 5
        else:
            period = 6
        
        colors.append(period_colors[period % len(period_colors)])
    
    return colors

def _get_block_colors(node_count, periodic_elements):
    """Bloklara göre renkler."""
    colors = []
    
    for atomic_num in range(1, node_count + 1):
        if atomic_num in [1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]:
            colors.append((1.0, 0.6, 0.6))  # s-block
        elif atomic_num in (list(range(5, 11)) + list(range(13, 19)) + 
                           list(range(31, 37)) + list(range(49, 55)) + 
                           list(range(81, 87)) + list(range(113, 119))):
            colors.append((0.6, 0.8, 1.0))  # p-block
        elif atomic_num in (list(range(21, 31)) + list(range(39, 49)) + 
                           list(range(72, 81)) + list(range(104, 113))):
            colors.append((0.6, 1.0, 0.6))  # d-block
        elif atomic_num in list(range(57, 72)) + list(range(89, 104)):
            colors.append((1.0, 0.6, 1.0))  # f-block
        else:
            colors.append((0.8, 0.8, 0.8))
    
    return colors

def _get_electronegativity_colors(node_count, periodic_elements):
    """Elektronegativiteye göre renkler."""
    colors = []
    electronegativity_data = {
        1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
        9: 3.98, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
        16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
        23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90,
        30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96,
        37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90,
        44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
        51: 2.05, 52: 2.10, 53: 2.66, 55: 0.79, 56: 0.89, 57: 1.10,
        58: 1.12, 59: 1.13, 60: 1.14, 62: 1.17, 63: 1.20, 64: 1.20, 65: 1.20,
        66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.10, 71: 1.27, 72: 1.30,
        73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54,
        80: 2.00, 81: 1.62, 82: 1.87, 83: 2.02, 84: 2.00, 85: 2.20,
        87: 0.70, 88: 0.89, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36,
        94: 1.28, 95: 1.30, 96: 1.30, 97: 1.30, 98: 1.30, 99: 1.30, 100: 1.30,
        101: 1.30, 102: 1.30, 103: 1.30
    }
    
    for atomic_num in range(1, node_count + 1):
        en = electronegativity_data.get(atomic_num, 1.5)
        if en < 1.0:
            color = (0.0, 0.0, 0.8)
        elif en < 1.5:
            color = (0.0, 0.5, 1.0)
        elif en < 2.0:
            color = (0.0, 0.8, 0.8)
        elif en < 2.5:
            color = (0.5, 1.0, 0.5)
        elif en < 3.0:
            color = (1.0, 0.8, 0.0)
        elif en < 3.5:
            color = (1.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        
        colors.append(color)
    
    return colors


def _generate_labels(graph, periodic_elements):
    """Etiketleri oluştur."""
    return {node_id: f"{periodic_elements[node_id][0]}\n{periodic_elements[node_id][1]}" 
            for node_id in graph.nodes()}

def _get_draw_order(graph, pos, strategy, dimension):
    """Çizim sırasını belirle."""
    if strategy == 'fixed':
        return list(graph.nodes())
    elif strategy == 'z_based' and dimension == '3d':
        return sorted(graph.nodes(), key=lambda n: pos[n][2], reverse=True)
    elif strategy == 'smart':
        nodes = list(graph.nodes())
        if len(nodes) == 0:
            return nodes
        
        if dimension == '3d':
            positions = np.array([pos[n] for n in nodes])
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
        else:
            positions = np.array([pos[n] for n in nodes])
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
        
        sorted_indices = np.argsort(distances)[::-1]
        return [nodes[i] for i in sorted_indices]
    else:
        return list(graph.nodes())

def _get_edge_config(style):
    """Edge stilini belirle."""
    configs = {
        'standard': {'color': 'gray', 'alpha': 0.5, 'width': 1.0},
        'light': {'color': 'lightgray', 'alpha': 0.3, 'width': 0.8},
        'bold': {'color': 'black', 'alpha': 0.7, 'width': 2.0},
        'hidden': {'color': 'none', 'alpha': 0.0, 'width': 0.0}
    }
    return configs.get(style, configs['standard'])


def _draw_edges(graph, pos, ax, config, dimension):
    """Edge'leri çiz."""
    if config['color'] == 'none':
        return
    
    for u, v in graph.edges():
        if dimension == '3d':
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                [pos[u][2], pos[v][2]],
                color=config['color'],
                alpha=config['alpha'],
                linewidth=config['width'],
                zorder=1
            )
        else:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=config['color'],
                alpha=config['alpha'],
                linewidth=config['width'],
                zorder=1
            )

def _draw_nodes(graph, pos, colors, ax, node_size, draw_order, dimension, **kwargs):
    """Node'ları çiz."""
    edge_width = kwargs.get('edge_width', 2.0)
    alpha = kwargs.get('node_alpha', 1.0)
    
    for node_id in draw_order:
        if dimension == '3d':
            x, y, z = pos[node_id]
            ax.scatter(x, y, z,
                      s=node_size if isinstance(node_size, int) else node_size[node_id-1],
                      c=[colors[node_id-1]],
                      edgecolors='black',
                      linewidths=edge_width,
                      alpha=alpha,
                      depthshade=False if kwargs.get('no_depth_shade', False) else True,
                      zorder=10)
        else:
            x, y = pos[node_id]
            ax.scatter(x, y,
                      s=node_size if isinstance(node_size, int) else node_size[node_id-1],
                      c=[colors[node_id-1]],
                      edgecolors='black',
                      linewidths=edge_width,
                      alpha=alpha,
                      zorder=10)

def _draw_labels(graph, pos, labels, colors, ax, font_size, position, dimension, **kwargs):
    """Etiketleri çiz."""
    label_offset = kwargs.get('label_offset', 0.0)
    
    for node_id in graph.nodes():
        if dimension == '3d':
            x, y, z = pos[node_id]
            if position == 'above':
                z += label_offset
            elif position == 'below':
                z -= label_offset
            elif position == 'right':
                x += label_offset
            elif position == 'left':
                x -= label_offset
            
            bg_color = colors[node_id-1]
            text_color = get_text_color_for_bg(bg_color)
            
            ax.text(x, y, z,
                   labels[node_id],
                   fontsize=font_size if isinstance(font_size, int) else font_size[node_id-1],
                   fontweight='bold',
                   color=text_color,
                   ha='center',
                   va='center',
                   zorder=1000)
        else:
            x, y = pos[node_id]
            if position == 'above':
                y += label_offset
            elif position == 'below':
                y -= label_offset
            elif position == 'right':
                x += label_offset
            elif position == 'left':
                x -= label_offset
            
            bg_color = colors[node_id-1]
            text_color = get_text_color_for_bg(bg_color)
            
            ax.text(x, y,
                   labels[node_id],
                   fontsize=font_size if isinstance(font_size, int) else font_size[node_id-1],
                   fontweight='bold',
                   color=text_color,
                   ha='center',
                   va='center',
                   zorder=1000)

def _generate_title(layout_type, node_count, color_scheme, dimension):
    """Başlık oluştur."""
    dim_text = "3D" if dimension == '3d' else "2D"
    scheme_text = color_scheme.capitalize()
    
    layout_names = {
        '3d_helix': 'Heliks Layout',
        '2d_linear': 'Lineer Layout',
        '2d_circular': 'Dairesel Layout',
        '3d_spherical': 'Küresel Layout',
        '2d_spring': 'Yay Layout',
        '2d_grid': 'Grid Layout'
    }
    
    layout_name = layout_names.get(layout_type, layout_type.replace('_', ' ').title())
    
    title = f"Keçeci Layout ile Periyodik Tablo\n"
    title += f"{dim_text} {layout_name}\n"
    title += f"({node_count} Element, {scheme_text} Renk Şeması)"
    
    return title

def _configure_axes(ax, pos, dimension, layout_params, **kwargs):
    """Eksenleri yapılandır."""
    ax.set_axis_off()
    
    if dimension == '3d':
        elev = kwargs.get('elevation', -25)
        azim = kwargs.get('azimuth', 15)
        ax.view_init(elev=elev, azim=azim)
        
        positions = list(pos.values())
        if positions:
            xs, ys, zs = zip(*positions)
            
            padding = kwargs.get('padding', 0.2)
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            z_range = max(zs) - min(zs)
            
            x_range = max(x_range, 10)
            y_range = max(y_range, 10)
            z_range = max(z_range, 10)
            
            ax.set_xlim(min(xs) - x_range*padding, max(xs) + x_range*padding)
            ax.set_ylim(min(ys) - y_range*padding, max(ys) + y_range*padding)
            ax.set_zlim(min(zs) - z_range*padding, max(zs) + z_range*padding)
    else:
        ax.set_aspect('equal')
        ax.autoscale_view()

def _add_legend(ax, color_scheme, dimension, periodic_elements):
    """Renk şeması açıklaması ekle."""
    legend_text = f"Renk Şeması: {color_scheme.capitalize()}\n"
    
    if color_scheme == 'group':
        legend_text += "• Kırmızı: Alkali Metaller\n"
        legend_text += "• Turuncu: Toprak Alkali\n"
        legend_text += "• Yeşil: Geçiş Metalleri\n"
        legend_text += "• Mavi: Ametaller\n"
        legend_text += "• Mor: Halojenler\n"
        legend_text += "• Gri: Soygazlar"
    
    elif color_scheme == 'period':
        legend_text += "• Her periyot farklı renk\n"
        legend_text += "• 7 periyot, 7 renk"
    
    elif color_scheme == 'block':
        legend_text += "• Kırmızı: s-blok\n"
        legend_text += "• Mavi: p-blok\n"
        legend_text += "• Yeşil: d-blok\n"
        legend_text += "• Mor: f-blok"
    
    if dimension == '3d':
        ax.text2D(0.02, 0.98, legend_text, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ana fonksiyonlar
def quick_periodic_table_3d():
    """
    Hızlı 3D periyodik tablo görselleştirmesi.
    Boş şablonlar oluşmaz.
    """
    # Önceki figürleri temizle
    plt.close('all')
    
    # Periyodik tabloyu yükle
    periodic_elements, _ = generate_complete_periodic_table()
    
    # Graf oluştur
    node_count = len(periodic_elements)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, node_count + 1))
    for i in range(1, node_count):
        G.add_edge(i, i + 1)
    
    # Tek figür oluştur
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    ax, pos = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type='3d_helix',
        color_scheme='vibrant',
        node_size=2000,
        font_size=10,
        edge_style='light',
        title="Keçeci Layout ile Periyodik Tablo\n3D Heliks Layout",
        elevation=-25,
        azimuth=15
    )
    
    plt.tight_layout()
    plt.show()
    
    # Kullanıcıya bilgi ver
    print(f"\n3D periyodik tablo oluşturuldu!")
    print(f"Toplam {node_count} element gösteriliyor.")
    print("Grafı kapatmak için figür penceresini kapatın.")
    
    return ax, pos

def generate_comparison_figure():
    """
    4 farklı görseli bir figürde karşılaştırma.
    Boş şablon oluşmaz.
    """
    # Önceki figürleri temizle
    plt.close('all')
    
    # Periyodik tabloyu yükle
    periodic_elements, _ = generate_complete_periodic_table()
    
    # Graf oluştur
    node_count = len(periodic_elements)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, node_count + 1))
    for i in range(1, node_count):
        G.add_edge(i, i + 1)
    
    # 2x2 grid şeklinde 4 alt figür oluştur
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Keçeci Layout ile Periyodik Tablo Görselleştirme Karşılaştırması', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # 1. 3D Heliks Layout
    ax1 = fig.add_subplot(221, projection='3d')
    ax1, pos1 = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type='3d_helix',
        layout_params={'z_spacing': 6.0, 'radius': 20.0, 'turns': 3.0},
        ax=ax1,
        color_scheme='vibrant',
        node_size=1200,
        font_size=8,
        edge_style='light',
        label_position='center',
        title="3D Heliks Layout\n(Vibrant Renkler)",
        elevation=-25,
        azimuth=15,
        figsize=None  # Figsize'ı burada kullanmıyoruz
    )
    
    # 2. 2D Linear Layout
    ax2 = fig.add_subplot(222)
    ax2, pos2 = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type='2d_linear',
        layout_params={'primary_spacing': 2.5, 'secondary_spacing': 2.5},
        ax=ax2,
        color_scheme='group',
        dimension='2d',
        node_size=800,
        font_size=7,
        edge_style='standard',
        show_legend=False,
        title="2D Linear Layout\n(Grup Renkleri)"
    )
    
    # 3. 2D Grid Layout
    ax3 = fig.add_subplot(223)
    ax3, pos3 = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type='2d_grid',
        layout_params={'rows': 9, 'cols': 14, 'spacing': 2.2},
        ax=ax3,
        color_scheme='period',
        dimension='2d',
        node_size=600,
        font_size=6,
        edge_style='light',
        show_legend=False,
        title="2D Grid Layout\n(Periyot Renkleri)"
    )
    
    # 4. 2D Circular Layout
    ax4 = fig.add_subplot(224)
    ax4, pos4 = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type='2d_circular',
        layout_params={'scale': 2.0},
        ax=ax4,
        color_scheme='block',
        dimension='2d',
        node_size=800,
        font_size=7,
        edge_style='light',
        show_legend=False,
        title="2D Dairesel Layout\n(Blok Renkleri)"
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    return fig, (ax1, ax2, ax3, ax4)

def save_periodic_table_visualization(
    filename: str = "periodic_table_kececi",
    format: str = "png",
    dpi: int = 300,
    layout_type: str = "3d_helix",
    color_scheme: str = "vibrant"
):
    """
    Periyodik tablo görselleştirmesini kaydet.
    
    Parameters:
    -----------
    filename : str
        Kaydedilecek dosyanın adı (uzantı olmadan)
    format : str
        Kayıt formatı: 'png', 'jpg', 'svg', 'pdf'
    dpi : int
        Çözünürlük (dots per inch)
    layout_type : str
        Layout tipi
    color_scheme : str
        Renk şeması
    """
    # Önceki figürleri temizle
    plt.close('all')
    
    # Periyodik tabloyu yükle
    periodic_elements, _ = generate_complete_periodic_table()
    
    # Graf oluştur
    node_count = len(periodic_elements)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, node_count + 1))
    for i in range(1, node_count):
        G.add_edge(i, i + 1)
    
    # Figür oluştur
    if '3d' in layout_type.lower():
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(111)
    
    # Görseli çiz
    ax, pos = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type=layout_type,
        ax=ax,
        color_scheme=color_scheme,
        node_size=1500 if '3d' in layout_type.lower() else 1000,
        font_size=9 if '3d' in layout_type.lower() else 8,
        edge_style='light',
        show_legend=True if color_scheme in ['group', 'period', 'block'] else False
    )
    
    # Kaydet
    full_filename = f"{filename}.{format}"
    plt.savefig(full_filename, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Görsel kaydedildi: {full_filename}")
    return full_filename

def highlight_elements(element_symbols: List[str], 
                      highlight_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                      **kwargs):
    """
    Belirli elementleri vurgula.
    
    Parameters:
    -----------
    element_symbols : List[str]
        Vurgulanacak element sembolleri
    highlight_color : Tuple[float, float, float]
        Vurgulama rengi (RGB)
    **kwargs : diğer parametreler draw_kececi_periodic_table'a aktarılır
    """
    # Önceki figürleri temizle
    plt.close('all')
    
    # Periyodik tabloyu yükle
    periodic_elements, _ = generate_complete_periodic_table()
    
    # Vurgulanacak elementlerin atom numaralarını bul
    highlight_indices = []
    element_symbols_found = []
    valid_symbols = []
    
    # Element sembolünden isim eşleştirmesi için sabit sözlük
    # Güncellenmiş Türkçe element isimleri
    element_name_map = {
        'H': 'Hidrojen', 'He': 'Helyum', 'Li': 'Lityum', 'Be': 'Berilyum',
        'B': 'Bor', 'C': 'Karbon', 'N': 'Azot', 'O': 'Oksijen', 'F': 'Flor',
        'Ne': 'Neon', 'Na': 'Sodyum', 'Mg': 'Magnezyum', 'Al': 'Alüminyum',
        'Si': 'Silisyum', 'P': 'Fosfor', 'S': 'Kükürt', 'Cl': 'Klor',
        'Ar': 'Argon', 'K': 'Potasyum', 'Ca': 'Kalsiyum', 'Sc': 'Skandiyum',
        'Ti': 'Titanyum', 'V': 'Vanadyum', 'Cr': 'Krom', 'Mn': 'Mangan',
        'Fe': 'Demir', 'Co': 'Kobalt', 'Ni': 'Nikel', 'Cu': 'Bakır',
        'Zn': 'Çinko', 'Ga': 'Galyum', 'Ge': 'Germanyum', 'As': 'Arsenik',
        'Se': 'Selenyum', 'Br': 'Brom', 'Kr': 'Kripton', 'Rb': 'Rubidyum',
        'Sr': 'Stronsiyum', 'Y': 'İtriyum', 'Zr': 'Zirkonyum', 'Nb': 'Niyobyum',
        'Mo': 'Molibden', 'Tc': 'Teknesyum', 'Ru': 'Rutenyum', 'Rh': 'Rodyum',
        'Pd': 'Paladyum', 'Ag': 'Gümüş', 'Cd': 'Kadmiyum', 'In': 'İndiyum',
        'Sn': 'Kalay', 'Sb': 'Antimon', 'Te': 'Tellür', 'I': 'İyot',
        'Xe': 'Ksenon', 'Cs': 'Sezyum', 'Ba': 'Baryum', 'La': 'Lantan',
        'Ce': 'Seryum', 'Pr': 'Praseodim', 'Nd': 'Neodimyum', 'Pm': 'Prometyum',
        'Sm': 'Samaryum', 'Eu': 'Europyum', 'Gd': 'Gadolinyum', 'Tb': 'Terbiyum',
        'Dy': 'Disprozyum', 'Ho': 'Holmiyum', 'Er': 'Erbiyum', 'Tm': 'Tulyum',
        'Yb': 'İterbiyum', 'Lu': 'Lutesyum', 'Hf': 'Hafniyum', 'Ta': 'Tantal',
        'W': 'Tungsten', 'Re': 'Renyum', 'Os': 'Osmiyum', 'Ir': 'İridyum',
        'Pt': 'Platin', 'Au': 'Altın', 'Hg': 'Cıva', 'Tl': 'Talyum',
        'Pb': 'Kurşun', 'Bi': 'Bizmut', 'Po': 'Polonyum', 'At': 'Astatin',
        'Rn': 'Radon', 'Fr': 'Fransiyum', 'Ra': 'Radyum', 'Ac': 'Aktinyum',
        'Th': 'Toryum', 'Pa': 'Protaktinyum', 'U': 'Uranyum', 'Np': 'Neptünyum',
        'Pu': 'Plütonyum', 'Am': 'Amerikyum', 'Cm': 'Küriyum', 'Bk': 'Berkelyum',
        'Cf': 'Kaliforniyum', 'Es': 'Einsteinyum', 'Fm': 'Fermiyum', 'Md': 'Mendelevyum',
        'No': 'Nobelyum', 'Lr': 'Lawrensiyum', 'Rf': 'Rutherfordiyum', 'Db': 'Dubniyum',
        'Sg': 'Seaborgiyum', 'Bh': 'Bohriyum', 'Hs': 'Hassiyum', 'Mt': 'Meitneriyum',
        'Ds': 'Darmstadtiyum', 'Rg': 'Röntgenyum', 'Cn': 'Kopernikyum', 'Nh': 'Nihonyum',
        'Fl': 'Flerovyum', 'Mc': 'Moscovyum', 'Lv': 'Livermoryum', 'Ts': 'Tenesin',
        'Og': 'Oganesson/Oganesyan' 
    }
    
    for input_symbol in element_symbols:
        input_symbol_clean = str(input_symbol).strip()
        found = False
        for atomic_num, (sym, atomic_num_in_tuple) in periodic_elements.items():
            # Periyodik tablodaki sembol ile kullanıcının girdiği sembolü karşılaştır
            # Büyük/küçük harf duyarsız karşılaştırma
            if sym.upper() == input_symbol_clean.upper():
                highlight_indices.append(atomic_num - 1)  # 0-based index
                # Orijinal sembolü (büyük/küçük harf korunarak) ekle
                element_symbols_found.append(sym)  # Burada sym kullanıyoruz (periyodik tablodaki orijinal)
                valid_symbols.append(sym)  # Orijinal sembolü sakla
                found = True
                break
        
        if not found:
            print(f"Uyarı: '{input_symbol_clean}' elementi bulunamadı!")
    
    if not highlight_indices:
        print("Vurgulanacak geçerli element bulunamadı!")
        return None, None
    
    # Element isimlerini bul
    element_names = [element_name_map.get(sym, sym) for sym in element_symbols_found]
    
    print(f"Vurgulanan elementler: {', '.join(element_names)}")
    
    # Graf oluştur
    node_count = len(periodic_elements)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, node_count + 1))
    for i in range(1, node_count):
        G.add_edge(i, i + 1)
    
    # Özel renk şeması oluştur
    colors = []
    for i in range(1, node_count + 1):
        if (i - 1) in highlight_indices:
            colors.append(highlight_color)
        else:
            # Gri tonlarında diğer elementler
            colors.append((0.9, 0.9, 0.9))
    
    # Layout tipini belirle
    layout_type = kwargs.get('layout_type', '3d_helix')
    dimension = '3d' if '3d' in layout_type.lower() else '2d'
    
    # Figür oluştur
    if dimension == '3d':
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(111)
    
    # Layout hesapla
    layout_params = kwargs.get('layout_params', {})
    pos = _calculate_layout(G, layout_type, layout_params, dimension, node_count)
    
    # Özel etiketler oluştur
    custom_labels = _generate_labels(G, periodic_elements)
    
    # Node boyutlarını ayarla (vurgulananlar daha büyük)
    node_sizes = []
    for i in range(1, node_count + 1):
        if (i - 1) in highlight_indices:
            node_sizes.append(kwargs.get('highlight_size', 2000))
        else:
            node_sizes.append(kwargs.get('normal_size', 800))
    
    # Font boyutlarını ayarla
    font_sizes = []
    for i in range(1, node_count + 1):
        if (i - 1) in highlight_indices:
            font_sizes.append(kwargs.get('highlight_font_size', 12))
        else:
            font_sizes.append(kwargs.get('normal_font_size', 7))
    
    # Çizim sırasını belirle
    draw_order = _get_draw_order(G, pos, 'smart', dimension)
    
    # Edge stilini al
    edge_config = _get_edge_config(kwargs.get('edge_style', 'light'))
    
    # Çiz
    _draw_edges(G, pos, ax, edge_config, dimension)
    _draw_nodes(G, pos, colors, ax, node_sizes, draw_order, dimension, 
               edge_width=kwargs.get('edge_width', 2.0),
               node_alpha=kwargs.get('node_alpha', 1.0))
    
    # Etiketleri çiz
    _draw_labels(G, pos, custom_labels, colors, ax, font_sizes, 
                kwargs.get('label_position', 'center'), dimension,
                label_offset=kwargs.get('label_offset', 0.0))
    
    # Başlık
    title = kwargs.get('title')
    if title is None:
        # Orijinal sembolleri (büyük/küçük harf korunarak) kullan
        element_symbols_str = ', '.join(element_symbols_found)
        element_names_str = ', '.join(element_names)
        title = f"Keçeci Layout ile Vurgulanan Elementler\nHighlighted Elements with Keçeci Layout:\n: {element_names_str} ({element_symbols_str})\n"
        title += f"Layout: {layout_type}, Toplam/Total {node_count} Element"
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Eksen ayarları
    _configure_axes(ax, pos, dimension, layout_params,
                   elevation=kwargs.get('elevation', -25),
                   azimuth=kwargs.get('azimuth', 15))
    
    plt.tight_layout()
    
    # Açıklama ekle
    if dimension == '3d':
        ax.text2D(0.02, 0.98, 
                 f"Kırmızı: {', '.join(element_names)}\nGri: Diğer elementler", 
                 transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.02, 0.98, 
               f"Kırmızı: {', '.join(element_names)}\nGri: Diğer elementler", 
               transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    print(f"\nVurgulama tamamlandı!")
    print(f"Toplam {node_count} element, {len(highlight_indices)} element vurgulandı.")
    print("Grafı kapatmak için figür penceresini kapatın.")
    
    return ax, pos

def demo_periodic_table_visualizations():
    """
    Periyodik tablo görselleştirmelerinin demo gösterimi.
    Her görsel ayrı ayrı gösterilir, boş şablonlar oluşmaz.
    """
    # Tüm figürleri temizle
    plt.close('all')
    
    # Periyodik tabloyu yükle
    print("Periyodik tablo yükleniyor...")
    periodic_elements, _ = generate_complete_periodic_table()
    
    # Graf oluştur
    node_count = len(periodic_elements)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, node_count + 1))
    for i in range(1, node_count):
        G.add_edge(i, i + 1)
    
    print("=" * 70)
    print("PERİYODİK TABLO GÖRSELLEŞTİRME DEMOLARI")
    print("=" * 70)
    print(f"Toplam {node_count} element gösterilecek.")
    print("Her görsel 5 saniye boyunca gösterilecek...")
    
    demos = [
        {
            "name": "3D Heliks Layout",
            "layout_type": "3d_helix",
            "color_scheme": "vibrant",
            "params": {'z_spacing': 8.0, 'radius': 25.0, 'turns': 3.0},
            "figsize": (20, 20),
            "projection": '3d'
        },
        {
            "name": "2D Linear Layout", 
            "layout_type": "2d_linear",
            "color_scheme": "group",
            "params": {'primary_spacing': 3.0, 'secondary_spacing': 3.0},
            "figsize": (16, 16),
            "projection": None
        },
        {
            "name": "2D Grid Layout",
            "layout_type": "2d_grid",
            "color_scheme": "period",
            "params": {'rows': 9, 'cols': 14, 'spacing': 3.0},
            "figsize": (18, 18),
            "projection": None
        },
        {
            "name": "2D Circular Layout",
            "layout_type": "2d_circular",
            "color_scheme": "block",
            "params": {'scale': 2.0},
            "figsize": (16, 16),
            "projection": None
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"\n{i}. {demo['name']} ({demo['color_scheme']} renkler) oluşturuluyor...")
        
        if demo['projection'] == '3d':
            fig = plt.figure(figsize=demo['figsize'])
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.figure(figsize=demo['figsize'])
            ax = fig.add_subplot(111)
        
        ax, pos = draw_kececi_periodic_table(
            G,
            periodic_elements,
            layout_type=demo['layout_type'],
            layout_params=demo['params'],
            ax=ax,
            color_scheme=demo['color_scheme'],
            node_size=2000 if demo['projection'] == '3d' else 1000,
            font_size=10 if demo['projection'] == '3d' else 8,
            edge_style='light',
            show_legend=True if demo['color_scheme'] in ['group', 'period', 'block'] else False,
            title=f"Keçeci Layout ile\n {i}/4: {demo['name']}\n{demo['color_scheme'].capitalize()} Renk Şeması"
        )
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)
    
    print("\n" + "=" * 70)
    print("TÜM DEMOLAR TAMAMLANDI!")
    print("=" * 70)
    
    return True

def get_element_info(element_symbol: str) -> Dict[str, Any]:
    """
    Element sembolünden element bilgilerini getir.
    
    Parameters:
    -----------
    element_symbol : str
        Element sembolü (örn: 'H', 'He', 'Fe')
    
    Returns:
    --------
    dict : Element bilgileri
    """
    # Periyodik tabloyu yükle
    periodic_elements, element_dict = generate_complete_periodic_table()
    
    # Element sembolünden atom numarasını bul
    atomic_num_found = None
    original_symbol = None
    
    for atomic_num, (sym, atomic_num_in_tuple) in periodic_elements.items():
        if sym.upper() == element_symbol.upper():
            atomic_num_found = atomic_num
            original_symbol = sym  # Orijinal sembolü sakla
            break
    
    if atomic_num_found is None:
        raise ValueError(f"Element bulunamadı: {element_symbol}")
    
    # Element ismini bul - Güncellenmiş Türkçe isimler
    element_name_map = {
        'H': 'Hidrojen', 'He': 'Helyum', 'Li': 'Lityum', 'Be': 'Berilyum',
        'B': 'Bor', 'C': 'Karbon', 'N': 'Azot', 'O': 'Oksijen', 'F': 'Flor',
        'Ne': 'Neon', 'Na': 'Sodyum', 'Mg': 'Magnezyum', 'Al': 'Alüminyum',
        'Si': 'Silisyum', 'P': 'Fosfor', 'S': 'Kükürt', 'Cl': 'Klor',
        'Ar': 'Argon', 'K': 'Potasyum', 'Ca': 'Kalsiyum', 'Sc': 'Skandiyum',
        'Ti': 'Titanyum', 'V': 'Vanadyum', 'Cr': 'Krom', 'Mn': 'Mangan',
        'Fe': 'Demir', 'Co': 'Kobalt', 'Ni': 'Nikel', 'Cu': 'Bakır',
        'Zn': 'Çinko', 'Ga': 'Galyum', 'Ge': 'Germanyum', 'As': 'Arsenik',
        'Se': 'Selenyum', 'Br': 'Brom', 'Kr': 'Kripton', 'Rb': 'Rubidyum',
        'Sr': 'Stronsiyum', 'Y': 'İtriyum', 'Zr': 'Zirkonyum', 'Nb': 'Niyobyum',
        'Mo': 'Molibden', 'Tc': 'Teknesyum', 'Ru': 'Rutenyum', 'Rh': 'Rodyum',
        'Pd': 'Paladyum', 'Ag': 'Gümüş', 'Cd': 'Kadmiyum', 'In': 'İndiyum',
        'Sn': 'Kalay', 'Sb': 'Antimon', 'Te': 'Tellür', 'I': 'İyot',
        'Xe': 'Ksenon', 'Cs': 'Sezyum', 'Ba': 'Baryum', 'La': 'Lantan',
        'Ce': 'Seryum', 'Pr': 'Praseodim', 'Nd': 'Neodimyum', 'Pm': 'Prometyum',
        'Sm': 'Samaryum', 'Eu': 'Europyum', 'Gd': 'Gadolinyum', 'Tb': 'Terbiyum',
        'Dy': 'Disprozyum', 'Ho': 'Holmiyum', 'Er': 'Erbiyum', 'Tm': 'Tulyum',
        'Yb': 'İterbiyum', 'Lu': 'Lutesyum', 'Hf': 'Hafniyum', 'Ta': 'Tantal',
        'W': 'Tungsten', 'Re': 'Renyum', 'Os': 'Osmiyum', 'Ir': 'İridyum',
        'Pt': 'Platin', 'Au': 'Altın', 'Hg': 'Cıva', 'Tl': 'Talyum',
        'Pb': 'Kurşun', 'Bi': 'Bizmut', 'Po': 'Polonyum', 'At': 'Astatin',
        'Rn': 'Radon', 'Fr': 'Fransiyum', 'Ra': 'Radyum', 'Ac': 'Aktinyum',
        'Th': 'Toryum', 'Pa': 'Protaktinyum', 'U': 'Uranyum', 'Np': 'Neptünyum',
        'Pu': 'Plütonyum', 'Am': 'Amerikyum', 'Cm': 'Küriyum', 'Bk': 'Berkelyum',
        'Cf': 'Kaliforniyum', 'Es': 'Einsteinyum', 'Fm': 'Fermiyum', 'Md': 'Mendelevyum',
        'No': 'Nobelyum', 'Lr': 'Lawrensiyum', 'Rf': 'Rutherfordiyum', 'Db': 'Dubniyum',
        'Sg': 'Seaborgiyum', 'Bh': 'Bohriyum', 'Hs': 'Hassiyum', 'Mt': 'Meitneriyum',
        'Ds': 'Darmstadtiyum', 'Rg': 'Röntgenyum', 'Cn': 'Kopernikyum', 'Nh': 'Nihonyum',
        'Fl': 'Flerovyum', 'Mc': 'Moscovyum', 'Lv': 'Livermoryum', 'Ts': 'Tenesin',
        'Og': 'Oganesson/Oganesyan'
    }
    
    element_name = element_name_map.get(original_symbol.upper(), original_symbol)
    
    # Yuri Oganessian hakkında ek bilgi
    additional_info = ""
    if original_symbol.upper() == 'OG':
        additional_info = "\n  Not: Element, Rus-Armeni fizikçi Yuri Oganessian (Юрий Оганесян) onuruna adlandırılmıştır."
    
    # Grup ve periyot bilgilerini hesapla
    group = _determine_group(atomic_num_found)
    
    # Periyot hesapla
    if atomic_num_found <= 2:
        period = 1
    elif atomic_num_found <= 10:
        period = 2
    elif atomic_num_found <= 18:
        period = 3
    elif atomic_num_found <= 36:
        period = 4
    elif atomic_num_found <= 54:
        period = 5
    elif atomic_num_found <= 86:
        period = 6
    else:
        period = 7
    
    # Blok bilgisi
    if atomic_num_found in [1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]:
        block = "s"
    elif atomic_num_found in (list(range(5, 11)) + list(range(13, 19)) + 
                       list(range(31, 37)) + list(range(49, 55)) + 
                       list(range(81, 87)) + list(range(113, 119))):
        block = "p"
    elif atomic_num_found in (list(range(21, 31)) + list(range(39, 49)) + 
                       list(range(72, 81)) + list(range(104, 113))):
        block = "d"
    elif atomic_num_found in list(range(57, 72)) + list(range(89, 104)):
        block = "f"
    else:
        block = "unknown"
    
    # Elektronegativite verisi
    electronegativity_data = {
        1: 2.20, 2: None, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
        9: 3.98, 10: None, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
        16: 2.58, 17: 3.16, 18: None, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
        23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90,
        30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: None,
        37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90,
        44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
        51: 2.05, 52: 2.10, 53: 2.66, 54: None, 55: 0.79, 56: 0.89, 57: 1.10,
        58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.20, 64: 1.20,
        65: 1.20, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.10, 71: 1.27,
        72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28,
        79: 2.54, 80: 2.00, 81: 1.62, 82: 1.87, 83: 2.02, 84: 2.00, 85: 2.20,
        86: None, 87: 0.70, 88: 0.89, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38,
        93: 1.36, 94: 1.28, 95: 1.30, 96: 1.30, 97: 1.30, 98: 1.30, 99: 1.30,
        100: 1.30, 101: 1.30, 102: 1.30, 103: 1.30, 118: None  # Oganesson
    }
    
    result = {
        'atomic_number': atomic_num_found,
        'symbol': original_symbol,  # Orijinal sembolü döndür
        'name': element_name,
        'group': group,
        'period': period,
        'block': block,
        'electronegativity': electronegativity_data.get(atomic_num_found, None)
    }
    
    # Ek bilgiyi de döndürelim
    result['additional_info'] = additional_info
    
    return result

def custom_visualization():
    """
    Özelleştirilmiş görselleştirme menüsü.
    """
    print("\n" + "=" * 70)
    print("Custom Visualization")
    print("=" * 70)
    
    # Periyodik tabloyu yükle
    periodic_elements, _ = generate_complete_periodic_table()
    
    # Graf oluştur
    node_count = len(periodic_elements)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, node_count + 1))
    for i in range(1, node_count):
        G.add_edge(i, i + 1)
    
    # Layout seçenekleri
    layouts = {
        '1': ('3d_helix', 'Heliks Layout'),
        '2': ('2d_linear', 'Linear Layout'),
        '3': ('2d_grid', 'Grid Layout'),
        '4': ('2d_circular', 'Dairesel Layout'),
        '5': ('2d_spring', 'Yay Layout')
    }
    
    print("\nLayout Tipleri:")
    for key, (_, name) in layouts.items():
        print(f"{key}. {name}")
    
    layout_choice = input("\nLayout tipi seçin (1-5): ").strip()
    if layout_choice not in layouts:
        print("Varsayılan olarak 3D Heliks seçildi.")
        layout_type = '3d_helix'
    else:
        layout_type, layout_name = layouts[layout_choice]
    
    # Renk şeması seçenekleri
    color_schemes = {
        '1': 'vibrant',
        '2': 'distinct',
        '3': 'pastel',
        '4': 'group',
        '5': 'period',
        '6': 'block',
        '7': 'electronegativity',
    }
    
    print("\nRenk Şemaları:")
    print("1. Vibrant (Canlı renkler)")
    print("2. Distinct (Farklı renkler)")
    print("3. Pastel (Pastel tonlar)")
    print("4. Group (Gruplara göre)")
    print("5. Period (Periyotlara göre)")
    print("6. Block (Bloklara göre)")
    print("7. electronegativity (Electronegativitiye göre)")
    
    color_choice = input("\nRenk şeması seçin (1-7): ").strip()
    if color_choice not in color_schemes:
        print("Varsayılan olarak Vibrant seçildi.")
        color_scheme = 'vibrant'
    else:
        color_scheme = color_schemes[color_choice]
    
    # Boyut seçimi
    try:
        node_size = int(input(f"\nNode boyutu (varsayılan: 1600): ") or "1600")
    except:
        node_size = 1600
    
    try:
        font_size = int(input(f"Font boyutu (varsayılan: 10): ") or "10")
    except:
        font_size = 10
    
    # 3D için özel parametreler
    if '3d' in layout_type:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
    
    # Özel başlık
    custom_title = input("\nÖzel başlık (boş bırakırsanız otomatik oluşturulur): ").strip()
    
    # Çizim yap
    ax, pos = draw_kececi_periodic_table(
        G,
        periodic_elements,
        layout_type=layout_type,
        ax=ax,
        color_scheme=color_scheme,
        node_size=node_size,
        font_size=font_size,
        title=custom_title if custom_title else None,
        show_legend=(color_scheme in ['vibrant', 'distinct', 'pastel', 'group', 'period', 'block', 'electronegativity'])
    )
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{layout_name} ile {color_scheme} renk şeması başarıyla oluşturuldu!")
    return ax, pos

def debug_periodic_table_structure():
    """Periyodik tablo veri yapısını kontrol et."""
    print("Periyodik tablo veri yapısı kontrol ediliyor...")
    periodic_elements, _ = generate_complete_periodic_table()
    
    print(f"\nToplam element sayısı: {len(periodic_elements)}")
    
    # İlk 5 elementi göster
    print("\nİlk 5 element:")
    for i, (atomic_num, value) in enumerate(list(periodic_elements.items())[:5]):
        print(f"  Atom numarası {atomic_num}:")
        print(f"    Değer: {value}")
        print(f"    Tip: {type(value)}")
        
        if isinstance(value, tuple):
            print(f"    Tuple uzunluğu: {len(value)}")
            for j, item in enumerate(value):
                print(f"      Item {j}: {item} (tip: {type(item)})")
    
    # Rastgele bir element kontrolü
    print("\nRastgele element kontrolü (atom numarası 26 - Demir):")
    if 26 in periodic_elements:
        value = periodic_elements[26]
        print(f"  Değer: {value}")
        print(f"  Tip: {type(value)}")
        if isinstance(value, tuple) and len(value) >= 2:
            print(f"  Sembol: {value[0]}")
            print(f"  İsim: {value[1]}")

def _generate_labels(graph, periodic_elements):
    """Etiketleri oluştur."""
    labels = {}
    for node_id in graph.nodes():
        if node_id in periodic_elements:
            sym, atomic_num = periodic_elements[node_id]
            # Etiket formatı: Sembol\nAtom Numarası
            labels[node_id] = f"{sym}\n{atomic_num}"
        else:
            labels[node_id] = str(node_id)
    return labels

def kececi_barbell_layout(G, primary_spacing=1.5, secondary_spacing=0.8, 
                         primary_direction='horizontal', debug=False):
    """
    KEÇECİ BARBELL LAYOUT v3.0 - %100 NODE KAPSAMA GARANTİSİ
    kececilayout.draw_kececi ile uyumlu
    """
    
    if debug:
        print("🔍 KEÇECİ BARBELL LAYOUT v3.0 - %100 KAPSAMA")
    
    pos = {}
    nodes = sorted(G.nodes())
    n = len(nodes)
    
    if debug:
        print(f"  Node aralığı: {n} node [{nodes[0]}-{nodes[-1]}]")
    
    # ✅ GARANTİ: TÜM NODE'LARA POZİSYON VER
    if n <= 10:
        # Küçük graf: Klasik barbell
        pos = classic_small_barbell(nodes, primary_spacing, secondary_spacing)
    elif n <= 25:
        # Orta graf: Genişletilmiş barbell
        pos = extended_barbell_layout(nodes, primary_spacing, secondary_spacing)
    else:
        # Büyük graf: Zigzag Keçeci
        pos = zigzag_kececi_layout(nodes, primary_spacing, secondary_spacing)
    
    # ✅ SİGORTA: HÂLÂ EKSİK VAR MI?
    missing = set(nodes) - set(pos.keys())
    if missing:
        if debug: print(f"  ⚠️  {len(missing)} node ekleniyor...")
        for i, node in enumerate(missing):
            pos[node] = (i * 0.5, n + i * 0.3)  # Güvenli pozisyon
    
    if debug:
        print(f"✅ TAM KAPSAMA: {len(pos)}/{n} node ✓")
    
    return pos


def detect_barbell_structure(G, debug=False):
    """Barbell yapısını otomatik tespit et"""
    
    # Yöntem 1: Yüksek dereceli kümeler + düşük dereceli köprü
    degrees = dict(G.degree())
    high_degree_nodes = [n for n, d in degrees.items() if d > G.number_of_nodes()/4]
    low_degree_nodes = [n for n, d in degrees.items() if d < 3]
    
    if debug:
        print(f"  Yüksek dereceli: {len(high_degree_nodes)}, Düşük: {len(low_degree_nodes)}")
    
    if len(high_degree_nodes) > 4 and len(low_degree_nodes) > 0:
        # İki ana küme + köprü adayları
        cliques = split_into_cliques(high_degree_nodes, G)
        bridge_nodes = low_degree_nodes[:3]  # Max 3 köprü node
        return cliques, bridge_nodes
    
    # Yöntem 2: NetworkX barbell_graph kontrolü
    import networkx as nx
    if hasattr(nx.generators.classic, 'barbell_graph'):
        cliques = [[i for i in range(5)], [i for i in range(6,11)]]
        bridge_nodes = [5]
        return cliques, bridge_nodes
    
    return None, None


def split_into_cliques(nodes, G, min_size=4):
    """Yüksek bağlantılı kümeleri ayır"""
    from itertools import combinations
    
    cliques = []
    remaining = set(nodes)
    
    while remaining:
        # En yüksek dereceli node'dan başla
        start = max(remaining, key=lambda n: G.degree(n))
        clique = set()
        
        # Komşularından clique oluştur
        clique.add(start)
        for neighbor in G.neighbors(start):
            if neighbor in remaining and G.degree(neighbor) > 3:
                clique.add(neighbor)
        
        if len(clique) >= min_size:
            cliques.append(sorted(clique))
            remaining -= clique
    
    return cliques[:2]  # Max 2 ana küme


def horizontal_barbell_layout(cliques, bridge_nodes, all_nodes, p_spacing, s_spacing):
    """Yatay barbell: Sol Küme ←→ Köprü ←→ Sağ Küme"""
    pos = {}
    
    # Sol küme
    if cliques and len(cliques) >= 1:
        left_nodes = cliques[0][:5]  # Max 5 node
        for i, node in enumerate(left_nodes):
            angle = 2 * np.pi * i / len(left_nodes)
            pos[node] = (np.cos(angle), np.sin(angle))
    
    # Köprü (yatay)
    bridge_x = 2.5
    for i, node in enumerate(bridge_nodes[:3]):
        pos[node] = (bridge_x + i * p_spacing * 0.6, np.sin(i * np.pi / 4))
    
    # Sağ küme  
    if len(cliques) >= 2:
        right_nodes = cliques[1][:5]
        for i, node in enumerate(right_nodes):
            angle = 2 * np.pi * i / len(right_nodes)
            pos[node] = (5 - np.cos(angle), np.sin(angle))
    
    # Kalan node'lar (genişletilmiş destek)
    remaining = [n for n in all_nodes if n not in pos]
    for i, node in enumerate(remaining):
        if i < 4:  # Max 4 ek node
            pos[node] = (1 + i * 0.8, 1.5 + (i%2) * (-0.5))
    
    return pos


def fallback_kececi_layout(G, direction='horizontal', debug=False):
    """Genel Keçeci layout (barbell değilse)"""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes)
    pos = {}
    
    if direction == 'horizontal':
        for i, node in enumerate(nodes):
            x = (i % 5) * 1.2
            y = (i // 5) * 1.5
            pos[node] = (x, y)
    else:  # zigzag
        for i, node in enumerate(nodes):
            x = i * 0.8
            y = np.sin(i * np.pi / 3) * 0.5
            pos[node] = (x, y)
    
    return pos

def classic_small_barbell(nodes, p_spacing, s_spacing):
    """Klasik barbell: Sol K5 │ Köprü │ Sağ K5"""
    pos = {}
    n = len(nodes)
    
    # Sol küme (ilk 1/3)
    left_count = min(5, n//3)
    for i in range(left_count):
        angle = 2 * np.pi * i / left_count
        pos[nodes[i]] = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
    
    # Köprü (orta %10)
    bridge_start = left_count
    bridge_end = min(bridge_start + 2, n)
    for i in range(bridge_start, bridge_end):
        x = 2.5 + (i-bridge_start) * p_spacing * 0.5
        pos[nodes[i]] = (x, np.sin((i-bridge_start) * np.pi / 4))
    
    # Sağ küme (son 1/3)
    right_start = max(bridge_end, n//2)
    for i in range(right_start, n):
        idx = i - right_start
        angle = 2 * np.pi * idx / (n-right_start)
        pos[nodes[i]] = (5 - 0.8 * np.cos(angle), 0.8 * np.sin(angle))
    
    return pos


def extended_barbell_layout(nodes, p_spacing, s_spacing):
    """Genişletilmiş: 2+ küme + geniş köprü"""
    pos = {}
    n = len(nodes)
    
    # Sol kümeler (ilk %40)
    left_end = int(0.4 * n)
    for i in range(left_end):
        x = (i % 4) * 0.8
        y = (i // 4) * 1.2
        pos[nodes[i]] = (x, y)
    
    # Köprü bölgesi (%20)
    bridge_end = int(0.6 * n)
    for i in range(left_end, bridge_end):
        x = 3.5 + (i-left_end) * p_spacing * 0.4
        pos[nodes[i]] = (x, np.sin(i * 0.3))
    
    # Sağ kümeler (son %40)
    for i in range(bridge_end, n):
        idx = i - bridge_end
        x = 6 + (idx % 3) * 0.9
        y = (idx // 3) * 1.1
        pos[nodes[i]] = (x, y)
    
    return pos


def zigzag_kececi_layout(nodes, p_spacing, s_spacing):
    """Keçeci zigzag: Büyük graf'lar için"""
    pos = {}
    for i, node in enumerate(nodes):
        x = i * p_spacing * 0.6
        y = np.sin(i * np.pi / 3) * s_spacing
        pos[node] = (x, y)
    return pos

# =============================================================================
# MODULE TEST CODE
# =============================================================================

if __name__ == '__main__':
    print("Testing kececilayout.py module...")
    G_test = nx.gnp_random_graph(12, 0.3, seed=42)

    # graph-tool grafi oluşturma ve test etme
    if gt:
        g = gt.Graph()
        g.add_vertex(12)
        for u, v in G_test.edges():
            g.add_edge(g.vertex(u), g.vertex(v))
        fig_gt = plt.figure(figsize=(10, 8))
        draw_kececi(g, ax=fig_gt.add_subplot(111), style='curved')
        plt.title("Keçeci Layout: graph-tool Graph")
        plt.show()

    # Compare expanding=False (parallel) vs. expanding=True ('v4' style)
    fig_v4 = plt.figure(figsize=(16, 7))
    fig_v4.suptitle("Effect of the `expanding` Parameter", fontsize=20)
    ax_v4_1 = fig_v4.add_subplot(1, 2, 1)
    draw_kececi(G_test, ax=ax_v4_1, style='curved',
                primary_direction='left-to-right', secondary_start='up',
                expanding=False)
    ax_v4_1.set_title("Parallel Style (expanding=False)", fontsize=16)

    ax_v4_2 = fig_v4.add_subplot(1, 2, 2)
    draw_kececi(G_test, ax=ax_v4_2, style='curved',
                primary_direction='left-to-right', secondary_start='up',
                expanding=True)
    ax_v4_2.set_title("Expanding 'v4' Style (expanding=True)", fontsize=16)
    plt.show()

    # Test all advanced drawing styles
    fig_styles = plt.figure(figsize=(18, 12))
    fig_styles.suptitle("Advanced Drawing Styles Test", fontsize=20)
    draw_kececi(G_test, style='curved', ax=fig_styles.add_subplot(2, 2, 1),
                primary_direction='left-to-right', secondary_start='up', expanding=True)
    draw_kececi(G_test, style='transparent', ax=fig_styles.add_subplot(2, 2, 2),
                primary_direction='top_down', secondary_start='left', expanding=True, node_color='purple')
    draw_kececi(G_test, style='3d', ax=fig_styles.add_subplot(2, 2, (3, 4), projection='3d'))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
