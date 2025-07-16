# kececilayout/kececi_layout.py

import itertools # Graphillion için eklendi
import numpy as np # rustworkx
import math
import networkx as nx
import rustworkx as rx
import igraph as ig
import networkit as nk
import graphillion as gg
import random


# Gerekli olabilecek kütüphane importları (type hinting veya isinstance için)
try:
    import networkx as nx
except ImportError:
    nx = None # Yoksa None ata
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
    import graphillion as gg # Graphillion importu eklendi
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

def kececi_layout_v4(graph, primary_spacing=1.0, secondary_spacing=1.0,
                     primary_direction='top-down', secondary_start='right'):
    """
    Keçeci Layout v4 - Graf düğümlerine sıralı-zigzag yerleşimi sağlar.
    NetworkX, Rustworkx, igraph, Networkit ve Graphillion grafikleriyle çalışır.

    Parametreler:
    -------------
    graph : Kütüphaneye özel graf nesnesi
        NetworkX, Rustworkx, igraph, Networkit veya Graphillion nesnesi.
        Graphillion için bir GraphSet nesnesi beklenir.
    ... (diğer parametreler) ...

    Dönüş:
    ------
    dict[node_identifier, tuple[float, float]]
        Her düğümün koordinatını içeren sözlük. Anahtarlar kütüphaneye
        göre değişir (NX: node obj/id, RW/NK/igraph: int index, GG: 1-based int index).
    """

    nodes = None

    # Kütüphane Tespiti ve Düğüm Listesi Çıkarımı
    # isinstance kullanmak hasattr'dan daha güvenilirdir.
    # Önce daha spesifik tipleri kontrol etmek iyi olabilir.

    if gg and isinstance(graph, gg.GraphSet): # Graphillion kontrolü EKLENDİ
        edges = graph.universe()
        max_node_id = find_max_node_id(edges)
        if max_node_id > 0:
            nodes = list(range(1, max_node_id + 1)) # 1-tabanlı indeksleme
        else:
            nodes = [] # Boş evren
        print(f"DEBUG: Graphillion tespit edildi. Düğümler (1..{max_node_id}): {nodes[:10]}...") # Debug mesajı

    elif ig and isinstance(graph, ig.Graph): # igraph
        nodes = sorted([v.index for v in graph.vs]) # 0-tabanlı indeksleme
        print(f"DEBUG: igraph tespit edildi. Düğümler (0..{len(nodes)-1}): {nodes[:10]}...")

    elif nk and isinstance(graph, nk.graph.Graph): # Networkit
        try:
            # iterNodes genellikle 0..N-1 verir ama garanti değil
            nodes = sorted(list(graph.iterNodes()))
        except Exception:
             nodes = list(range(graph.numberOfNodes()))
        print(f"DEBUG: Networkit tespit edildi. Düğümler: {nodes[:10]}...")

    elif rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):  # Rustworkx (hem yönlü hem yönsüz)
        nodes = sorted(graph.node_indices()) # 0-tabanlı indeksleme
        print(f"DEBUG: Rustworkx tespit edildi. Düğümler (0..{len(nodes)-1}): {nodes[:10]}...")

    elif nx and isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)): # NetworkX
        try:
            # Düğümler sıralanabilirse sırala (genelde int/str)
            nodes = sorted(list(graph.nodes()))
        except TypeError:
             # Sıralanamayan düğüm tipleri varsa (örn. tuple, nesne) sırasız al
             nodes = list(graph.nodes())
        print(f"DEBUG: NetworkX tespit edildi. Düğümler: {nodes[:10]}...")

    else:
        # Desteklenmeyen tip veya kütüphane kurulu değilse
        supported_types = []
        if nx: supported_types.append("NetworkX")
        if rx: supported_types.append("Rustworkx")
        if ig: supported_types.append("igraph")
        if nk: supported_types.append("Networkit")
        if gg: supported_types.append("Graphillion.GraphSet")
        raise TypeError(f"Unsupported graph type: {type(graph)}. Desteklenen türler: {', '.join(supported_types)}")

    # ----- Buradan sonrası tüm kütüphaneler için ortak -----

    num_nodes = len(nodes)
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

    # Ana döngü - Düğümleri sıralı indekslerine göre yerleştirir,
    # sözlüğe ise gerçek düğüm ID/indeks/nesnesini anahtar olarak kullanır.
    for i, node_id in enumerate(nodes):
        # i: Düğümün sıralı listedeki 0-tabanlı indeksi (0, 1, 2, ...) - Yerleşim için kullanılır
        # node_id: Gerçek düğüm kimliği/indeksi - Sonuç sözlüğünün anahtarı

        # 1. Ana eksen koordinatını hesapla
        if primary_direction == 'top-down':
            primary_coord = i * -primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing; secondary_axis = 'y'
        else: # right-to-left
            primary_coord = i * -primary_spacing; secondary_axis = 'y'

        # 2. Yan eksen ofsetini hesapla (zigzag)
        if i == 0: secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        # 3. (x, y) koordinatlarını ata
        if secondary_axis == 'x': x, y = secondary_coord, primary_coord
        else: x, y = primary_coord, secondary_coord

        # Sonuç sözlüğüne ekle
        pos[node_id] = (x, y)

    return pos
 
def kececi_layout(graph, primary_spacing=1.0, secondary_spacing=1.0,
                     primary_direction='top-down', secondary_start='right'):
    """
    Keçeci Layout v4 - Graf düğümlerine sıralı-zigzag yerleşimi sağlar.
    NetworkX, Rustworkx, igraph, Networkit ve Graphillion grafikleriyle çalışır.

    Parametreler:
    -------------
    graph : Kütüphaneye özel graf nesnesi
        NetworkX, Rustworkx, igraph, Networkit veya Graphillion nesnesi.
        Graphillion için bir GraphSet nesnesi beklenir.
    ... (diğer parametreler) ...

    Dönüş:
    ------
    dict[node_identifier, tuple[float, float]]
        Her düğümün koordinatını içeren sözlük. Anahtarlar kütüphaneye
        göre değişir (NX: node obj/id, RW/NK/igraph: int index, GG: 1-based int index).
    """

    nodes = None

    # Kütüphane Tespiti ve Düğüm Listesi Çıkarımı
    # isinstance kullanmak hasattr'dan daha güvenilirdir.
    # Önce daha spesifik tipleri kontrol etmek iyi olabilir.

    if gg and isinstance(graph, gg.GraphSet): # Graphillion kontrolü EKLENDİ
        edges = graph.universe()
        max_node_id = find_max_node_id(edges)
        if max_node_id > 0:
            nodes = list(range(1, max_node_id + 1)) # 1-tabanlı indeksleme
        else:
            nodes = [] # Boş evren
        print(f"DEBUG: Graphillion tespit edildi. Düğümler (1..{max_node_id}): {nodes[:10]}...") # Debug mesajı

    elif ig and isinstance(graph, ig.Graph): # igraph
        nodes = sorted([v.index for v in graph.vs]) # 0-tabanlı indeksleme
        print(f"DEBUG: igraph tespit edildi. Düğümler (0..{len(nodes)-1}): {nodes[:10]}...")

    elif nk and isinstance(graph, nk.graph.Graph): # Networkit
        try:
            # iterNodes genellikle 0..N-1 verir ama garanti değil
            nodes = sorted(list(graph.iterNodes()))
        except Exception:
             nodes = list(range(graph.numberOfNodes()))
        print(f"DEBUG: Networkit tespit edildi. Düğümler: {nodes[:10]}...")

    elif rx and isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):  # Rustworkx (hem yönlü hem yönsüz)
        nodes = sorted(graph.node_indices()) # 0-tabanlı indeksleme
        print(f"DEBUG: Rustworkx tespit edildi. Düğümler (0..{len(nodes)-1}): {nodes[:10]}...")

    elif nx and isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)): # NetworkX
        try:
            # Düğümler sıralanabilirse sırala (genelde int/str)
            nodes = sorted(list(graph.nodes()))
        except TypeError:
             # Sıralanamayan düğüm tipleri varsa (örn. tuple, nesne) sırasız al
             nodes = list(graph.nodes())
        print(f"DEBUG: NetworkX tespit edildi. Düğümler: {nodes[:10]}...")

    else:
        # Desteklenmeyen tip veya kütüphane kurulu değilse
        supported_types = []
        if nx: supported_types.append("NetworkX")
        if rx: supported_types.append("Rustworkx")
        if ig: supported_types.append("igraph")
        if nk: supported_types.append("Networkit")
        if gg: supported_types.append("Graphillion.GraphSet")
        raise TypeError(f"Unsupported graph type: {type(graph)}. Desteklenen türler: {', '.join(supported_types)}")

    # ----- Buradan sonrası tüm kütüphaneler için ortak -----

    num_nodes = len(nodes)
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

    # Ana döngü - Düğümleri sıralı indekslerine göre yerleştirir,
    # sözlüğe ise gerçek düğüm ID/indeks/nesnesini anahtar olarak kullanır.
    for i, node_id in enumerate(nodes):
        # i: Düğümün sıralı listedeki 0-tabanlı indeksi (0, 1, 2, ...) - Yerleşim için kullanılır
        # node_id: Gerçek düğüm kimliği/indeksi - Sonuç sözlüğünün anahtarı

        # 1. Ana eksen koordinatını hesapla
        if primary_direction == 'top-down':
            primary_coord = i * -primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing; secondary_axis = 'y'
        else: # right-to-left
            primary_coord = i * -primary_spacing; secondary_axis = 'y'

        # 2. Yan eksen ofsetini hesapla (zigzag)
        if i == 0: secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        # 3. (x, y) koordinatlarını ata
        if secondary_axis == 'x': x, y = secondary_coord, primary_coord
        else: x, y = primary_coord, secondary_coord

        # Sonuç sözlüğüne ekle
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
    if num_nodes == 0: return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_id in enumerate(nodes):
        # 1. Ana Eksen Koordinatını Hesapla
        if primary_direction == 'top-down': primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': primary_coord, secondary_axis = i * primary_spacing, 'y'
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
    if num_nodes == 0: return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_id in enumerate(nodes):
        # 1. Ana Eksen Koordinatını Hesapla
        if primary_direction == 'top-down': primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': primary_coord, secondary_axis = i * primary_spacing, 'y'
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
            primary_coord = i * -primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing; secondary_axis = 'y'
        else: # right-to-left
            primary_coord = i * -primary_spacing; secondary_axis = 'y'

        if i == 0: secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        if secondary_axis == 'x': x, y = secondary_coord, primary_coord
        else: x, y = primary_coord, secondary_coord
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
            primary_coord = i * -primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'bottom-up':
            primary_coord = i * primary_spacing; secondary_axis = 'x'
        elif primary_direction == 'left-to-right':
            primary_coord = i * primary_spacing; secondary_axis = 'y'
        else: # right-to-left
            primary_coord = i * -primary_spacing; secondary_axis = 'y'

        if i == 0: secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        if secondary_axis == 'x': x, y = secondary_coord, primary_coord
        else: x, y = primary_coord, secondary_coord
        pos[node_id] = (x, y)

    return pos

def kececi_layout_v4_rx(graph: rx.PyGraph, primary_spacing=1.0, secondary_spacing=1.0,
                        primary_direction='top-down', secondary_start='right'):
    pos = {}
    nodes = sorted(graph.node_indices())
    num_nodes = len(nodes)
    if num_nodes == 0: return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_index in enumerate(nodes):
        if primary_direction == 'top-down': primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: primary_coord, secondary_axis = i * -primary_spacing, 'y'

        if i == 0: secondary_offset_multiplier = 0.0
        else:
            start_mult = 1.0 if secondary_start in ['right', 'up'] else -1.0
            magnitude = math.ceil(i / 2.0)
            side = 1 if i % 2 != 0 else -1
            secondary_offset_multiplier = start_mult * magnitude * side
        secondary_coord = secondary_offset_multiplier * secondary_spacing

        x, y = (secondary_coord, primary_coord) if secondary_axis == 'x' else (primary_coord, secondary_coord)
        pos[node_index] = np.array([x, y])
    return pos

def kececi_layout_v4_rustworkx(graph: rx.PyGraph, primary_spacing=1.0, secondary_spacing=1.0,
                        primary_direction='top-down', secondary_start='right'):
    pos = {}
    nodes = sorted(graph.node_indices())
    num_nodes = len(nodes)
    if num_nodes == 0: return {}

    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: raise ValueError(f"Invalid secondary_start for vertical: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: raise ValueError(f"Invalid secondary_start for horizontal: {secondary_start}")

    for i, node_index in enumerate(nodes):
        if primary_direction == 'top-down': primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: primary_coord, secondary_axis = i * -primary_spacing, 'y'

        if i == 0: secondary_offset_multiplier = 0.0
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
    if min_nodes < 2: min_nodes = 2
    if max_nodes < min_nodes: max_nodes = min_nodes
    while True:
        num_nodes_target = random.randint(min_nodes, max_nodes)
        edge_probability = random.uniform(edge_prob_min, edge_prob_max)
        G_candidate = rx.PyGraph()
        node_indices = G_candidate.add_nodes_from([None] * num_nodes_target)
        for i in range(num_nodes_target):
            for j in range(i + 1, num_nodes_target):
                if random.random() < edge_probability:
                    G_candidate.add_edge(node_indices[i], node_indices[j], None)

        if G_candidate.num_nodes() == 0: continue
        if num_nodes_target > 1 and G_candidate.num_edges() == 0: continue

        if not rx.is_connected(G_candidate):
             components = rx.connected_components(G_candidate)
             if not components: continue
             largest_cc_nodes_indices = max(components, key=len, default=set())
             if len(largest_cc_nodes_indices) < 2 and num_nodes_target >=2 : continue
             if not largest_cc_nodes_indices: continue
             # Set'i listeye çevirerek subgraph oluştur
             G = G_candidate.subgraph(list(largest_cc_nodes_indices))
             if G.num_nodes() == 0: continue
        else:
             G = G_candidate

        if G.num_nodes() >= 2: break
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
    if num_nodes == 0: return {}
    is_vertical = primary_direction in ['top-down', 'bottom-up']
    is_horizontal = primary_direction in ['left-to-right', 'right-to-left']
    if not (is_vertical or is_horizontal): raise ValueError(f"Invalid primary_direction: {primary_direction}")
    if is_vertical and secondary_start not in ['right', 'left']: raise ValueError(f"Dikey yön için geçersiz secondary_start: {secondary_start}")
    if is_horizontal and secondary_start not in ['up', 'down']: raise ValueError(f"Yatay yön için geçersiz secondary_start: {secondary_start}")

    for i, node_id in enumerate(sorted_nodes):
        primary_coord = 0.0
        secondary_axis = ''
        if primary_direction == 'top-down': primary_coord, secondary_axis = i * -primary_spacing, 'x'
        elif primary_direction == 'bottom-up': primary_coord, secondary_axis = i * primary_spacing, 'x'
        elif primary_direction == 'left-to-right': primary_coord, secondary_axis = i * primary_spacing, 'y'
        else: primary_coord, secondary_axis = i * -primary_spacing, 'y'

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

    if min_nodes < 2: min_nodes = 2
    if max_nodes < min_nodes: max_nodes = min_nodes
    while True:
        num_nodes_target = random.randint(min_nodes, max_nodes)
        edge_probability = random.uniform(edge_prob_min, edge_prob_max)
        G_candidate = nx.gnp_random_graph(num_nodes_target, edge_probability, seed=None)
        if G_candidate.number_of_nodes() == 0: continue
        # Düzeltme: 0 kenarlı ama >1 düğümlü grafı da tekrar dene
        if num_nodes_target > 1 and G_candidate.number_of_edges() == 0 : continue

        if not nx.is_connected(G_candidate):
            # Düzeltme: default=set() kullanmak yerine önce kontrol et
            connected_components = list(nx.connected_components(G_candidate))
            if not connected_components: continue # Bileşen yoksa tekrar dene
            largest_cc_nodes = max(connected_components, key=len)
            if len(largest_cc_nodes) < 2 and num_nodes_target >=2 : continue
            if not largest_cc_nodes: continue # Bu aslında gereksiz ama garanti olsun
            G = G_candidate.subgraph(largest_cc_nodes).copy()
            if G.number_of_nodes() == 0: continue
        else: G = G_candidate
        if G.number_of_nodes() >= 2: break
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    print(f"Oluşturulan Graf: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Kenar (Başlangıç p={edge_probability:.3f})")
    return G

def generate_random_graph_ig(min_nodes=0, max_nodes=200, edge_prob_min=0.15, edge_prob_max=0.4):
    """igraph kullanarak rastgele bağlı bir graf oluşturur."""

    if min_nodes < 2: min_nodes = 2
    if max_nodes < min_nodes: max_nodes = min_nodes
    while True:
        num_nodes_target = random.randint(min_nodes, max_nodes)
        edge_probability = random.uniform(edge_prob_min, edge_prob_max)
        g_candidate = ig.Graph.Erdos_Renyi(n=num_nodes_target, p=edge_probability, directed=False)
        if g_candidate.vcount() == 0: continue
        if num_nodes_target > 1 and g_candidate.ecount() == 0 : continue
        if not g_candidate.is_connected(mode='weak'):
            components = g_candidate.components(mode='weak')
            if not components or len(components) == 0: continue
            largest_cc_subgraph = components.giant()
            if largest_cc_subgraph.vcount() < 2 and num_nodes_target >=2 : continue
            g = largest_cc_subgraph
            if g.vcount() == 0: continue
        else: g = g_candidate
        if g.vcount() >= 2: break
    print(f"Oluşturulan igraph Graf: {g.vcount()} Düğüm, {g.ecount()} Kenar (Başlangıç p={edge_probability:.3f})")
    g.vs["label"] = [str(i) for i in range(g.vcount())]
    g.vs["degree"] = g.degree()
    return g
