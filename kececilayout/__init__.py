# __init__.py
# Bu dosya paketin başlangıç noktası olarak çalışır.
# Alt modülleri yükler, sürüm bilgileri tanımlar ve geriye dönük uyumluluk için uyarılar sağlar.

from __future__ import annotations
import importlib
import warnings
import os
# if os.getenv("DEVELOPMENT") == "true":
    # importlib.reload(kececi_layout) # F821 undefined name 'kececi_layout'

# Dışa aktarılacak semboller listesi
__all__ = [
    'kececi_layout_v4',
    'kececi_layout',
    'kececi_layout_v4_nx',
    'kececi_layout_v4_networkx',
    'kececi_layout_v4_ig',
    'kececi_layout_v4_igraph',
    'kececi_layout_v4_nk',
    'kececi_layout_v4_networkit',
    'kececi_layout_v4_gg',
    'kececi_layout_v4_graphillion',
    'kececi_layout_v4_rx',
    'kececi_layout_v4_rustworkx',
    'generate_random_rx_graph',
    'kececi_layout_v4_pure',
    'generate_random_graph',
    'generate_random_graph_ig'
]

# Göreli modül içe aktarmaları
# F401 hatasını önlemek için sadece kullanacağınız şeyleri dışa aktarın
# Aksi halde linter'lar "imported but unused" uyarısı verir
try:
    #from .kececi_layout import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    #from . import kececi_layout  # Modülün kendisine doğrudan erişim isteniyorsa
    from .kececi_layout import (
        kececi_layout_v4, 
        kececi_layout, 
        kececi_layout_v4_nx, 
        kececi_layout_v4_networkx, 
        kececi_layout_v4_ig, 
        kececi_layout_v4_igraph, 
        kececi_layout_v4_nk, 
        kececi_layout_v4_networkit, 
        kececi_layout_v4_gg,
        kececi_layout_v4_graphillion,
        kececi_layout_v4_rx,
        kececi_layout_v4_rustworkx,
        generate_random_rx_graph,
        kececi_layout_v4_pure,
        generate_random_graph,
        generate_random_graph_ig
    )
except ImportError as e:
    warnings.warn(f"Gerekli modül yüklenemedi: {e}", ImportWarning)

# Eski bir fonksiyonun yer tutucusu - gelecekte kaldırılacak
def eski_fonksiyon():
    """
    Kaldırılması planlanan eski bir fonksiyondur.
    Lütfen alternatif fonksiyonları kullanın.
    """
    warnings.warn(
        "eski_fonksiyon() artık kullanılmamaktadır ve gelecekte kaldırılacaktır. "
        "Lütfen yeni alternatif fonksiyonları kullanın. "
        "Keçeci Layout; Python 3.7-3.14 sürümlerinde sorunsuz çalışmalıdır.",
        category=DeprecationWarning,
        stacklevel=2
    )

# Paket sürüm numarası
__version__ = "0.2.5"


