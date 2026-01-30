# -*- coding: utf-8 -*-
# ruff: noqa: N806, N815
"""
kececilayout.py

This module provides sequential-zigzag ("Keçeci Layout") and advanced visualization styles for various Python graph libraries.
Bu modül, çeşitli Python graf kütüphaneleri için sıralı-zigzag ("Keçeci Layout") ve gelişmiş görselleştirme stilleri sağlar.
"""

from collections import defaultdict
import graphillion as gg
import igraph as ig
import itertools # Graphillion için eklendi
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkit as nk
import networkx as nx
import numpy as np # rustworkx
from numba import jit
import platform # graph_tool için
import random
import rustworkx as rx
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
        expanding (bool): If True (default), the zigzag offset grows, creating the
                          triangle-like 'v4' style. If False, the offset is constant,
                          creating parallel lines.

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

    # Create coordinate list (will be ordered by vertex IDs 0 to N-1)
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

    # Create coordinate list (will be ordered by vertex IDs 0 to N-1)
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
    # Create the node ID list: 1, 2, ..., num_vertices
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
    # Create the node ID list: 1, 2, ..., num_vertices
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
        return pos_3d
    
    total_angle = 2 * np.pi * turns
    for i, node_id in enumerate(nodes):
        z = i * z_spacing
        angle = (i / (total_nodes - 1)) * total_angle if total_nodes > 1 else 0
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        pos_3d[node_id] = (x, y, z)
    return pos_3d

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
        # Tamamen rastgele ton (hue)
        hue = random.random() # 0.0 - 1.0 arası
        # Soft görünüm için doygunluk (saturation) orta seviyede
        saturation = 0.4 + (random.random() * 0.4) # 0.4 - 0.8 arası
        # Soft görünüm için parlaklık (value) yüksek
        value = 0.7 + (random.random() * 0.3)     # 0.7 - 1.0 arası
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
    """
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
    """
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
            provided, a new figure and axis are created.
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


