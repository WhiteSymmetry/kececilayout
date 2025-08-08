# __init__.py

"""
kececilayout - A Python package for sequential-zigzag graph layouts
and advanced visualizations compatible with multiple graph libraries.
"""

from __future__ import annotations
import inspect
import warnings

# Paket sürüm numarası
__version__ = "0.2.8"

# =============================================================================
# OTOMATİK İÇE AKTARMA VE __all__ OLUŞTURMA
# Bu bölüm, yeni fonksiyon eklediğinizde elle güncelleme yapma
# ihtiyacını ortadan kaldırır.
# =============================================================================

# Ana modülümüzü içe aktarıyoruz
from . import kececi_layout

# __all__ listesini dinamik olarak dolduracağız
__all__ = []

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


