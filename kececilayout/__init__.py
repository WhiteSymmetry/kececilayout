# __init__.py

"""
kececilayout - A Python package for sequential-zigzag graph layouts
and advanced visualizations compatible with multiple graph libraries.
"""

from __future__ import annotations
import inspect
import warnings

# Paket sürüm numarası
__version__ = "0.4.6"

# =============================================================================
# OTOMATİK İÇE AKTARMA VE __all__ OLUŞTURMA
# Bu bölüm, yeni fonksiyon eklediğinizde elle güncelleme yapma
# ihtiyacını ortadan kaldırır.
# =============================================================================

# Ana modülümüzü içe aktarıyoruz
#from . import kececi_layout

from .kececi_layout import (  # Veya fonksiyonların bulunduğu asıl modül
    kececi_layout,
    draw_kececi,
    _draw_internal,  # Private fonksiyonu açıkça import edin
    _kececi_layout_3d_helix,
    kececi_layout_3d_helix_parametric,
    kececi_layout_v4,
    
    # Library-specific layout functions
    kececi_layout_nx,
    kececi_layout_networkx,
    kececi_layout_ig,
    kececi_layout_igraph,
    kececi_layout_nk,
    kececi_layout_networkit,
    kececi_layout_gg,
    kececi_layout_graphillion,
    kececi_layout_rx,
    kececi_layout_rustworkx,
    kececi_layout_pure,
    load_element_data_and_spectral_lines,
    wavelength_to_rgb,
    get_text_color_for_bg,
    generate_soft_random_colors,
    generate_distinct_colors,
    
    # Drawing functions
    draw_kececi,
    #'_draw_internal',  # <- TESTLER İÇİN GEREKLİ
    
    # Utility functions
    find_max_node_id,
    to_networkx,
    
    # Graph generation functions
    generate_random_graph,
    generate_random_graph_ig,
    generate_random_rx_graph
)

# __all__ listesini dinamik olarak dolduracağız
__all__ = [
    # Core layout functions
    'kececi_layout',
    'kececi_layout_v4',
    
    # Library-specific layout functions
    'kececi_layout_nx',
    'kececi_layout_networkx',
    'kececi_layout_ig',
    'kececi_layout_igraph',
    'kececi_layout_nk',
    'kececi_layout_networkit',
    'kececi_layout_gg',
    'kececi_layout_graphillion',
    'kececi_layout_rx',
    'kececi_layout_rustworkx',
    'kececi_layout_pure',
    'load_element_data_and_spectral_lines',
    'wavelength_to_rgb',
    'get_text_color_for_bg',
    'generate_soft_random_colors',
    'generate_distinct_colors',
    
    # Drawing functions
    'draw_kececi',
    '_draw_internal',  # <- TESTLER İÇİN GEREKLİ
    '_kececi_layout_3d_helix',
    'kececi_layout_3d_helix_parametric',
    
    # Utility functions
    'find_max_node_id',
    'to_networkx',
    
    # Graph generation functions
    'generate_random_graph',
    'generate_random_graph_ig',
    'generate_random_rx_graph'
]

# kececi_layout modülünün içindeki tüm üyelere (fonksiyonlar, sınıflar vb.) bak
for name, member in inspect.getmembers(kececi_layout):
    # Eğer üye bir fonksiyonsa VE adı '_' ile başlamıyorsa (yani public ise)
    if inspect.isfunction(member) and not name.startswith('_'):
        # Onu paketin ana seviyesine taşı (örn: kl.draw_kececi)
        globals()[name] = member
        # Ve dışa aktarılacaklar listesine ekle
        __all__.append(name)

# Temizlik: Döngüde kullanılan geçici değişkenleri sil
del inspect, name, member

# =============================================================================
# GERİYE DÖNÜK UYUMLULUK VE UYARILAR
# =============================================================================

def old_function_placeholder():
    """
    This is an old function scheduled for removal.
    Please use alternative functions.
    """
    warnings.warn(
        (
            "old_function_placeholder() is deprecated and will be removed in a future version. "
            "Please use the new alternative functions. "
            "Keçeci Layout should work smoothly on Python 3.7-3.14."
        ),
        category=DeprecationWarning,
        stacklevel=2
    )

# Eğer bu eski fonksiyonu da dışa aktarmak istiyorsanız, __all__ listesine ekleyin
# __all__.append('old_function_placeholder')










