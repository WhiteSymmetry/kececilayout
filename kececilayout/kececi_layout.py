# -*- coding: utf-8 -*-
# ruff: noqa: N806, N815
"""
kececilayout.py

Bu modül, çeşitli Python graf kütüphaneleri için sıralı-zigzag ("Keçeci Layout")
ve gelişmiş görselleştirme stilleri sağlar.
"""

import graphillion as gg
import igraph as ig
import itertools # Graphillion için eklendi
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkit as nk
import networkx as nx
import numpy as np # rustworkx
import random
import rustworkx as rx
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


def find_max_node_id(edges):
    """Verilen kenar listesindeki en büyük düğüm ID'sini bulur."""
    if not edges:
        return 0
    try:
      # Tüm düğüm ID'lerini tek bir kümede topla ve en büyüğünü bul
      all_nodes = set(itertools.chain.from_iterable(edges))
      return max(all_nodes) if all_nodes else 0
    except TypeError: # Eğer kenarlar (node, node) formatında değilse
      print("Uyarı: Kenar formatı beklenenden farklı, max node ID 0 varsayıldı.")
      return 0


def kececi_layout(graph, primary_spacing=1.0, secondary_spacing=1.0,
                  primary_direction='top_down', secondary_start='right',
                  expanding=True):
    """
    Calculates 2D sequential-zigzag coordinates for the nodes of a graph.

    This function is compatible with graphs from NetworkX, Rustworkx, igraph,
    Networkit, and Graphillion.

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
    # Bu blok, farklı kütüphanelerden düğüm listelerini doğru şekilde alır.
    nx_graph = to_networkx(graph) # Emin olmak için en başta dönüştür
    try:
        nodes = sorted(list(nx_graph.nodes()))
    except TypeError:
        nodes = list(nx_graph.nodes())

    pos = {}
    
    # --- DOĞRULANMIŞ KONTROL BLOĞU ---
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: '{primary_direction}'")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical direction: '{secondary_start}'")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal direction: '{secondary_start}'")
    # --- BİTİŞ ---

    for i, node_id in enumerate(nodes):
        primary_coord, secondary_axis = 0.0, ''
        if primary_direction == 'top-down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up':
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

        x, y = ((secondary_offset, primary_coord) if secondary_axis == 'x' else
                (primary_coord, secondary_offset))
        pos[node_id] = (x, y)
    return pos

# =============================================================================
# 1. TEMEL LAYOUT HESAPLAMA FONKSİYONU (2D)
# Bu fonksiyon sadece koordinatları hesaplar, çizim yapmaz.
# 1. LAYOUT CALCULATION FUNCTION (UNIFIED AND IMPROVED)
# =============================================================================

def kececi_layout_v4(graph, primary_spacing=1.0, secondary_spacing=1.0,
                  primary_direction='top_down', secondary_start='right',
                  expanding=True): # v4 davranışını kontrol etmek için parametre eklendi
    """
    Calculates 2D sequential-zigzag coordinates for the nodes of a graph.

    This function is compatible with graphs from NetworkX, Rustworkx, igraph,
    Networkit, and Graphillion.

    Args:
        graph: A graph object from a supported library.
        primary_spacing (float): The distance between nodes along the primary axis.
        secondary_spacing (float): The base unit for the zigzag offset.
        primary_direction (str): 'top_down', 'bottom_up', 'left_to_right', 'right_to_left'.
        secondary_start (str): Initial direction for the zigzag ('up', 'down', 'left', 'right').
        expanding (bool): If True (default), the zigzag offset grows, creating the
                          triangle-like 'v4' style. If False, the offset is constant,
                          creating parallel lines.

    Returns:
        dict: A dictionary of positions formatted as {node_id: (x, y)}.
    """
    # ==========================================================
    # Sizin orijinal, çoklu kütüphane uyumluluk bloğunuz burada korunuyor.
    # Bu, kodun sağlamlığını garanti eder.
    # ==========================================================
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
        except TypeError:
            nodes = list(graph.nodes())
    else:
        supported = ["NetworkX", "Rustworkx", "igraph", "Networkit", "Graphillion"]
        raise TypeError(f"Unsupported graph type: {type(graph)}. Supported: {', '.join(supported)}")
    # ==========================================================

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
        if primary_direction == 'top-down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom_up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else:  # 'right_to_left'
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        secondary_offset = 0.0
        if i > 0:
            start_multiplier = 1.0 if secondary_start in ['right', 'up'] else -1.0
            
            # --- YENİ ESNEK MANTIK BURADA ---
            # `expanding` True ise 'v4' stili gibi genişler, değilse sabit kalır.
            magnitude = math.ceil(i / 2.0) if expanding else 1.0
            
            side = 1 if i % 2 != 0 else -1
            secondary_offset = start_multiplier * magnitude * side * secondary_spacing

        x, y = ((secondary_offset, primary_coord) if secondary_axis == 'x' else
                (primary_coord, secondary_offset))
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_nx(graph, primary_spacing=1.0, secondary_spacing=1.0,
                     primary_direction='top-down', secondary_start='right'):
    """
    Genişletilmiş Keçeci Düzeni: Ana eksen boyunca ilerler, ikincil eksende artan şekilde sapar.
    Düğümler ikincil eksende daha geniş bir alana yayılır.
    """
    pos = {}
    # NetworkX 2.x ve 3.x uyumluluğu için listeye çevirme
    nodes = sorted(list(graph.nodes()))
    num_nodes = len(nodes)
    if num_nodes == 0: 
        return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): 
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: 
        raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: 
        raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_id in enumerate(nodes):
        # 1. Ana Eksen Koordinatını Hesapla
        if primary_direction == 'top-down': 
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': 
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': 
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. İkincil Eksen Koordinatını Hesapla (Genişletilmiş Sapma)
        if i == 0:
            secondary_offset_multiplier = 0.0
        else:
            # Sapma yönünü belirle (sağ/yukarı +1, sol/aşağı -1)
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            # Sapma büyüklüğünü belirle (i arttıkça artar: 1, 1, 2, 2, 3, 3, ...)
            magnitude = math.ceil(i / 2.0)
            # Sapma tarafını belirle (tek i için pozitif, çift i için negatif)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side

        secondary_coord = secondary_offset_multiplier * secondary_spacing

        # 3. (x, y) Koordinatlarını Ata
        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_networkx(graph, primary_spacing=1.0, secondary_spacing=1.0,
                     primary_direction='top-down', secondary_start='right'):
    """
    Genişletilmiş Keçeci Düzeni: Ana eksen boyunca ilerler, ikincil eksende artan şekilde sapar.
    Düğümler ikincil eksende daha geniş bir alana yayılır.
    """
    pos = {}
    # NetworkX 2.x ve 3.x uyumluluğu için listeye çevirme
    nodes = sorted(list(graph.nodes()))
    num_nodes = len(nodes)
    if num_nodes == 0: 
        return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): 
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: 
        raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: 
        raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_id in enumerate(nodes):
        # 1. Ana Eksen Koordinatını Hesapla
        if primary_direction == 'top-down': 
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': 
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': 
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        # 2. İkincil Eksen Koordinatını Hesapla (Genişletilmiş Sapma)
        if i == 0:
            secondary_offset_multiplier = 0.0
        else:
            # Sapma yönünü belirle (sağ/yukarı +1, sol/aşağı -1)
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            # Sapma büyüklüğünü belirle (i arttıkça artar: 1, 1, 2, 2, 3, 3, ...)
            magnitude = math.ceil(i / 2.0)
            # Sapma tarafını belirle (tek i için pozitif, çift i için negatif)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side

        secondary_coord = secondary_offset_multiplier * secondary_spacing

        # 3. (x, y) Koordinatlarını Ata
        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos[node_id] = (x, y)

    return pos


def kececi_layout_v4_ig(graph: ig.Graph, primary_spacing=1.0, secondary_spacing=1.0,
                            primary_direction='top-down', secondary_start='right'):
    """igraph.Graph nesnesi için Keçeci layout.

    Args:
        graph: igraph.Graph nesnesi.
        primary_spacing: Ana eksendeki düğümler arasındaki boşluk.
        secondary_spacing: İkincil eksendeki ofset boşluğu.
        primary_direction: Ana eksenin yönü ('top-down', 'bottom-up', 'left-to-right', 'right-to-left').
        secondary_start: İkincil eksendeki ilk ofsetin yönü ('right', 'left', 'up', 'down').

    Returns:
        Vertex ID'lerine göre sıralanmış koordinatların listesi (ör: [[x0,y0], [x1,y1], ...]).
    """
    num_nodes = graph.vcount()
    if num_nodes == 0:
        return []

    # Koordinat listesi oluştur (vertex ID'leri 0'dan N-1'e sıralı olacak şekilde)
    pos_list = [[0.0, 0.0]] * num_nodes
    # Vertex ID'leri zaten 0'dan N-1'e olduğu için doğrudan range kullanabiliriz
    nodes = range(num_nodes) # Vertex ID'leri

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i in nodes: # i burada vertex index'i (0, 1, 2...)
        if primary_direction == 'top-down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        if i == 0:
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side

        secondary_coord = secondary_offset_multiplier * secondary_spacing

        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos_list[i] = [x, y] # Listeye doğru index'e [x, y] olarak ekle

    # igraph Layout nesnesi gibi davranması için basit bir nesne döndürelim
    # veya doğrudan koordinat listesini kullanalım. Çizim fonksiyonu listeyi kabul eder.
    # return ig.Layout(pos_list) # İsterseniz Layout nesnesi de döndürebilirsiniz
    return pos_list # Doğrudan liste döndürmek en yaygın ve esnek yoldur

def kececi_layout_v4_igraph(graph: ig.Graph, primary_spacing=1.0, secondary_spacing=1.0,
                            primary_direction='top-down', secondary_start='right'):
    """igraph.Graph nesnesi için Keçeci layout.

    Args:
        graph: igraph.Graph nesnesi.
        primary_spacing: Ana eksendeki düğümler arasındaki boşluk.
        secondary_spacing: İkincil eksendeki ofset boşluğu.
        primary_direction: Ana eksenin yönü ('top-down', 'bottom-up', 'left-to-right', 'right-to-left').
        secondary_start: İkincil eksendeki ilk ofsetin yönü ('right', 'left', 'up', 'down').

    Returns:
        Vertex ID'lerine göre sıralanmış koordinatların listesi (ör: [[x0,y0], [x1,y1], ...]).
    """
    num_nodes = graph.vcount()
    if num_nodes == 0:
        return []

    # Koordinat listesi oluştur (vertex ID'leri 0'dan N-1'e sıralı olacak şekilde)
    pos_list = [[0.0, 0.0]] * num_nodes
    # Vertex ID'leri zaten 0'dan N-1'e olduğu için doğrudan range kullanabiliriz
    nodes = range(num_nodes) # Vertex ID'leri

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']:
        raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i in nodes: # i burada vertex index'i (0, 1, 2...)
        if primary_direction == 'top-down':
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up':
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right':
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: # right-to-left
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        if i == 0:
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side

        secondary_coord = secondary_offset_multiplier * secondary_spacing

        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos_list[i] = [x, y] # Listeye doğru index'e [x, y] olarak ekle

    # igraph Layout nesnesi gibi davranması için basit bir nesne döndürelim
    # veya doğrudan koordinat listesini kullanalım. Çizim fonksiyonu listeyi kabul eder.
    # return ig.Layout(pos_list) # İsterseniz Layout nesnesi de döndürebilirsiniz
    return pos_list # Doğrudan liste döndürmek en yaygın ve esnek yoldur

def kececi_layout_v4_nk(graph: nk.graph.Graph,
                               primary_spacing=1.0,
                               secondary_spacing=1.0,
                               primary_direction='top-down',
                               secondary_start='right'):
    """
    Keçeci Layout v4 - Networkit graf düğümlerine sıralı-zigzag yerleşimi sağlar.

    Parametreler:
    -------------
    graph : networkit.graph.Graph
        Kenar ve düğüm bilgisi içeren Networkit graf nesnesi.
    primary_spacing : float
        Ana yön mesafesi.
    secondary_spacing : float
        Yan yön mesafesi.
    primary_direction : str
        'top-down', 'bottom-up', 'left-to-right', 'right-to-left'.
    secondary_start : str
        Başlangıç yönü ('right', 'left', 'up', 'down').

    Dönüş:
    ------
    dict[int, tuple[float, float]]
        Her düğüm ID'sinin (Networkit'te genelde integer olur)
        koordinatını içeren sözlük.
    """

    # Networkit'te düğüm ID'leri genellikle 0'dan N-1'e sıralıdır,
    # ancak garantiye almak için sıralı bir liste alalım.
    # iterNodes() düğüm ID'lerini döndürür.
    try:
        # Networkit'te node ID'lerinin contiguous (0..n-1) olup olmadığını kontrol edebiliriz
        # ama her zaman böyle olmayabilir. iterNodes en genel yöntem.
        nodes = sorted(list(graph.iterNodes()))
    except Exception as e:
        print(f"Networkit düğüm listesi alınırken hata: {e}")
        # Alternatif olarak, eğer ID'lerin 0'dan başladığı varsayılıyorsa:
        # nodes = list(range(graph.numberOfNodes()))
        # Ancak iterNodes daha güvenlidir.
        return {} # Hata durumunda boş dön

    num_nodes = len(nodes) # Veya graph.numberOfNodes()
    if num_nodes == 0:
        return {}  # Boş graf için boş sözlük döndür

    pos = {}  # Sonuç sözlüğü
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    # Parametre kontrolleri
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for vertical primary_direction ('{primary_direction}'). Use 'right' or 'left'.")
    if is_horizontal and secondary_start not in ['up', 'down']:
         raise ValueError(f"Invalid secondary_start ('{secondary_start}') for horizontal primary_direction ('{primary_direction}'). Use 'up' or 'down'.")

    # Ana döngü
    for i, node_id in enumerate(nodes):
        # i: Düğümün sıralı listedeki indeksi (0, 1, 2, ...) - Yerleşim için kullanılır
        # node_id: Gerçek Networkit düğüm ID'si - Sonuç sözlüğünün anahtarı

        # 1. Ana eksen koordinatını hesapla
        if primary_direction == 'top-down':
            primary_coord = i * -primary_spacing
            secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing
            secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing
            secondary_axis = 'y'
        else: # primary_direction == 'right-to-left'
            primary_coord = i * -primary_spacing
            secondary_axis = 'y'

        # 2. Yan eksen ofsetini hesapla (zigzag)
        if i == 0:
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side

        secondary_coord = secondary_offset_multiplier * secondary_spacing

        # 3. (x, y) koordinatlarını ata
        if secondary_axis == 'x':
            x, y = secondary_coord, primary_coord
        else: # secondary_axis == 'y'
            x, y = primary_coord, secondary_coord

        # Sonuç sözlüğüne ekle: anahtar=düğüm ID, değer=(x, y) tuple'ı
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_networkit(graph: nk.graph.Graph,
                               primary_spacing=1.0,
                               secondary_spacing=1.0,
                               primary_direction='top-down',
                               secondary_start='right'):
    """
    Keçeci Layout v4 - Networkit graf düğümlerine sıralı-zigzag yerleşimi sağlar.

    Parametreler:
    -------------
    graph : networkit.graph.Graph
        Kenar ve düğüm bilgisi içeren Networkit graf nesnesi.
    primary_spacing : float
        Ana yön mesafesi.
    secondary_spacing : float
        Yan yön mesafesi.
    primary_direction : str
        'top-down', 'bottom-up', 'left-to-right', 'right-to-left'.
    secondary_start : str
        Başlangıç yönü ('right', 'left', 'up', 'down').

    Dönüş:
    ------
    dict[int, tuple[float, float]]
        Her düğüm ID'sinin (Networkit'te genelde integer olur)
        koordinatını içeren sözlük.
    """

    # Networkit'te düğüm ID'leri genellikle 0'dan N-1'e sıralıdır,
    # ancak garantiye almak için sıralı bir liste alalım.
    # iterNodes() düğüm ID'lerini döndürür.
    try:
        # Networkit'te node ID'lerinin contiguous (0..n-1) olup olmadığını kontrol edebiliriz
        # ama her zaman böyle olmayabilir. iterNodes en genel yöntem.
        nodes = sorted(list(graph.iterNodes()))
    except Exception as e:
        print(f"Networkit düğüm listesi alınırken hata: {e}")
        # Alternatif olarak, eğer ID'lerin 0'dan başladığı varsayılıyorsa:
        # nodes = list(range(graph.numberOfNodes()))
        # Ancak iterNodes daha güvenlidir.
        return {} # Hata durumunda boş dön

    num_nodes = len(nodes) # Veya graph.numberOfNodes()
    if num_nodes == 0:
        return {}  # Boş graf için boş sözlük döndür

    pos = {}  # Sonuç sözlüğü
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    # Parametre kontrolleri
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for vertical primary_direction ('{primary_direction}'). Use 'right' or 'left'.")
    if is_horizontal and secondary_start not in ['up', 'down']:
         raise ValueError(f"Invalid secondary_start ('{secondary_start}') for horizontal primary_direction ('{primary_direction}'). Use 'up' or 'down'.")

    # Ana döngü
    for i, node_id in enumerate(nodes):
        # i: Düğümün sıralı listedeki indeksi (0, 1, 2, ...) - Yerleşim için kullanılır
        # node_id: Gerçek Networkit düğüm ID'si - Sonuç sözlüğünün anahtarı

        # 1. Ana eksen koordinatını hesapla
        if primary_direction == 'top-down':
            primary_coord = i * -primary_spacing
            secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing
            secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing
            secondary_axis = 'y'
        else: # primary_direction == 'right-to-left'
            primary_coord = i * -primary_spacing
            secondary_axis = 'y'

        # 2. Yan eksen ofsetini hesapla (zigzag)
        if i == 0:
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side

        secondary_coord = secondary_offset_multiplier * secondary_spacing

        # 3. (x, y) koordinatlarını ata
        if secondary_axis == 'x':
            x, y = secondary_coord, primary_coord
        else: # secondary_axis == 'y'
            x, y = primary_coord, secondary_coord

        # Sonuç sözlüğüne ekle: anahtar=düğüm ID, değer=(x, y) tuple'ı
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_gg(graph_set: gg.GraphSet,
                                 primary_spacing=1.0,
                                 secondary_spacing=1.0,
                                 primary_direction='top-down',
                                 secondary_start='right'):
    """
    Keçeci Layout v4 - Graphillion evren grafının düğümlerine
    sıralı-zigzag yerleşimi sağlar.
    """

    # DÜZELTME: Evrenden kenar listesini al
    edges_in_universe = graph_set.universe()

    # DÜZELTME: Düğüm sayısını kenarlardan türet
    num_vertices = find_max_node_id(edges_in_universe)

    if num_vertices == 0:
        return {}

    # Graphillion genellikle 1-tabanlı düğüm indekslemesi kullanır.
    # Düğüm ID listesini oluştur: 1, 2, ..., num_vertices
    nodes = list(range(1, num_vertices + 1)) # En yüksek ID'ye kadar tüm nodları varsay

    pos = {}  # Sonuç sözlüğü
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    # Parametre kontrolleri (değişiklik yok)
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for vertical primary_direction ('{primary_direction}'). Use 'right' or 'left'.")
    if is_horizontal and secondary_start not in ['up', 'down']:
         raise ValueError(f"Invalid secondary_start ('{secondary_start}') for horizontal primary_direction ('{primary_direction}'). Use 'up' or 'down'.")

    # Ana döngü (değişiklik yok)
    for i, node_id in enumerate(nodes):
        # ... (Koordinat hesaplama kısmı aynı kalır) ...
        if primary_direction == 'top-down':
            primary_coord = i * -primary_spacing; 
            secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing; 
            secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing; 
            secondary_axis = 'y'
        else: # right-to-left
            primary_coord = i * -primary_spacing; 
            secondary_axis = 'y'

        if i == 0: 
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        if secondary_axis == 'x': 
            x, y = secondary_coord, primary_coord
        else: 
            x, y = primary_coord, secondary_coord
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_graphillion(graph_set: gg.GraphSet,
                                 primary_spacing=1.0,
                                 secondary_spacing=1.0,
                                 primary_direction='top-down',
                                 secondary_start='right'):
    """
    Keçeci Layout v4 - Graphillion evren grafının düğümlerine
    sıralı-zigzag yerleşimi sağlar.
    """

    # DÜZELTME: Evrenden kenar listesini al
    edges_in_universe = graph_set.universe()

    # DÜZELTME: Düğüm sayısını kenarlardan türet
    num_vertices = find_max_node_id(edges_in_universe)

    if num_vertices == 0:
        return {}

    # Graphillion genellikle 1-tabanlı düğüm indekslemesi kullanır.
    # Düğüm ID listesini oluştur: 1, 2, ..., num_vertices
    nodes = list(range(1, num_vertices + 1)) # En yüksek ID'ye kadar tüm nodları varsay

    pos = {}  # Sonuç sözlüğü
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']

    # Parametre kontrolleri (değişiklik yok)
    if not (is_vertical or is_horizontal):
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']:
        raise ValueError(f"Invalid secondary_start ('{secondary_start}') for vertical primary_direction ('{primary_direction}'). Use 'right' or 'left'.")
    if is_horizontal and secondary_start not in ['up', 'down']:
         raise ValueError(f"Invalid secondary_start ('{secondary_start}') for horizontal primary_direction ('{primary_direction}'). Use 'up' or 'down'.")

    # Ana döngü (değişiklik yok)
    for i, node_id in enumerate(nodes):
        # ... (Koordinat hesaplama kısmı aynı kalır) ...
        if primary_direction == 'top-down':
            primary_coord = i * -primary_spacing; 
            secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing; 
            secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing; 
            secondary_axis = 'y'
        else: # right-to-left
            primary_coord = i * -primary_spacing; 
            secondary_axis = 'y'

        if i == 0: 
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        if secondary_axis == 'x': 
            x, y = secondary_coord, primary_coord
        else: 
            x, y = primary_coord, secondary_coord
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_rx(graph: 
                        rx.PyGraph, primary_spacing=1.0, secondary_spacing=1.0,
                        primary_direction='top-down', secondary_start='right'):
    pos = {}
    nodes = sorted(graph.node_indices())
    num_nodes = len(nodes)
    if num_nodes == 0: 
        return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): 
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: 
        raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: 
        raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_index in enumerate(nodes):
        if primary_direction == 'top-down': 
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': 
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': 
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: 
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        if i == 0: 
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos[node_index] = np.array([x, y])
    return pos

def kececi_layout_v4_rustworkx(graph: 
                               rx.PyGraph, primary_spacing=1.0, secondary_spacing=1.0,
                        primary_direction='top-down', secondary_start='right'):
    pos = {}
    nodes = sorted(graph.node_indices())
    num_nodes = len(nodes)
    if num_nodes == 0: 
        return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): 
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: 
        raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: 
        raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_index in enumerate(nodes):
        if primary_direction == 'top-down': 
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': 
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': 
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: 
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        if i == 0: 
            secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos[node_index] = np.array([x, y])
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


def kececi_layout_v4_pure(nodes, primary_spacing=1.0, secondary_spacing=1.0,
                              primary_direction='top-down', secondary_start='right'):
    """
    Keçeci layout mantığını kullanarak düğüm pozisyonlarını hesaplar.
    Sadece standart Python ve math modülünü kullanır.
    """
    pos = {}
    # Tutarlı sıra garantisi için düğümleri sırala
    # Girdi zaten liste/tuple olsa bile kopyasını oluşturup sırala
    # ... (Bir önceki cevaptaki fonksiyonun TAMAMI buraya yapıştırılacak) ...
    try:
        sorted_nodes = sorted(list(nodes))
    except TypeError:
        print("Uyarı: Düğümler sıralanamadı...")
        sorted_nodes = list(nodes)

    num_nodes = len(sorted_nodes)
    if num_nodes == 0: 
        return {}
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): 
        raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: 
        raise ValueError(f"Dikey yön için geçersiz secondary_start: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: 
        raise ValueError(f"Yatay yön için geçersiz secondary_start: {secondary_start}")

    for i, node_id in enumerate(sorted_nodes):
        primary_coord = 0.0
        secondary_axis = ''
        if primary_direction == 'top-down': 
            primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': 
            primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': 
            primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: 
            primary_coord, secondary_axis = i * -primary_spacing, 'y'

        secondary_offset_multiplier = 0.0
        if i > 0:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos[node_id] = (x, y)
    return pos

# =============================================================================
# Rastgele Graf Oluşturma Fonksiyonu (NetworkX) - Değişiklik yok
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
    if rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        nx_graph.add_nodes_from(graph.node_indices())
        nx_graph.add_edges_from(graph.edge_list())
    elif ig and isinstance(graph, ig.Graph):
        nx_graph.add_nodes_from(v.index for v in graph.vs)
        nx_graph.add_edges_from(graph.get_edgelist())
    elif nk and isinstance(graph, nk.graph.Graph):
        nx_graph.add_nodes_from(graph.iterNodes())
        nx_graph.add_edges_from(graph.iterEdges())
    elif gg and isinstance(graph, gg.GraphSet):
        edges = graph.universe()
        max_node_id = max(set(itertools.chain.from_iterable(edges))) if edges else 0
        if max_node_id > 0:
            nx_graph.add_nodes_from(range(1, max_node_id + 1))
            nx_graph.add_edges_from(edges)
    else:
        # This block is rarely reached as _get_nodes_from_graph would fail first
        raise TypeError(f"Unsupported graph type {type(graph)} could not be converted to NetworkX.")
        #raise TypeError(f"Desteklenmeyen graf tipi {type(graph)} NetworkX'e dönüştürülemedi.")
    return nx_graph


def _kececi_layout_3d_helix(nx_graph):
    """Internal function: Arranges nodes in a helix along the Z-axis."""
    pos_3d = {}
    nodes = sorted(list(nx_graph.nodes()))
    for i, node_id in enumerate(nodes):
        angle, radius, z_step = i * (np.pi / 2.5), 1.0, i * 0.8
        pos_3d[node_id] = (np.cos(angle) * radius, np.sin(angle) * radius, z_step)
    return pos_3d


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


# =============================================================================
# 4. MAIN USER-FACING DRAWING FUNCTION
# =============================================================================

def draw_kececi(graph, style='curved', ax=None, **kwargs):
    """
    Draws a graph using the Keçeci Layout with a specified style.

    This function automatically handles graphs from different libraries
    (NetworkX, Rustworkx, igraph, etc.).

    Args:
        graph: The graph object to be drawn.
        style (str): The drawing style. Options: 'curved', 'transparent', '3d'.
        ax (matplotlib.axis.Axis, optional): The axis to draw on. If not
            provided, a new figure and axis are created.
        **kwargs: Additional keyword arguments passed to both `kececi_layout`
                  and the drawing functions (e.g., expanding=True, node_size=500).

    Returns:
        matplotlib.axis.Axis: The axis object where the graph was drawn.
    """
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


# =============================================================================
# MODULE TEST CODE
# =============================================================================

if __name__ == '__main__':
    print("Testing kececilayout.py module...")
    G_test = nx.gnp_random_graph(12, 0.3, seed=42)

    # Compare expanding=False (parallel) vs. expanding=True ('v4' style)
    fig_v4 = plt.figure(figsize=(16, 7))
    fig_v4.suptitle("Effect of the `expanding` Parameter", fontsize=20)
    ax_v4_1 = fig_v4.add_subplot(1, 2, 1)
    draw_kececi(G_test, ax=ax_v4_1, style='curved',
                primary_direction='left_to_right', secondary_start='up',
                expanding=False)
    ax_v4_1.set_title("Parallel Style (expanding=False)", fontsize=16)

    ax_v4_2 = fig_v4.add_subplot(1, 2, 2)
    draw_kececi(G_test, ax=ax_v4_2, style='curved',
                primary_direction='left_to_right', secondary_start='up',
                expanding=True)
    ax_v4_2.set_title("Expanding 'v4' Style (expanding=True)", fontsize=16)
    plt.show()

    # Test all advanced drawing styles
    fig_styles = plt.figure(figsize=(18, 12))
    fig_styles.suptitle("Advanced Drawing Styles Test", fontsize=20)
    draw_kececi(G_test, style='curved', ax=fig_styles.add_subplot(2, 2, 1),
                primary_direction='left_to_right', secondary_start='up', expanding=True)
    draw_kececi(G_test, style='transparent', ax=fig_styles.add_subplot(2, 2, 2),
                primary_direction='top_down', secondary_start='left', expanding=True, node_color='purple')
    draw_kececi(G_test, style='3d', ax=fig_styles.add_subplot(2, 2, (3, 4), projection='3d'))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

