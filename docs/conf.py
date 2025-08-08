# Configuration file for the Sphinx documentation builder.
import os
import sys

# Proje kök dizinini Python yoluna ekle (kececilayout'u bulmak için)
sys.path.insert(0, os.path.abspath('..'))

# Proje Bilgileri
project = 'kececilayout'
copyright = '2025, Mehmet Keçeci'
author = 'Mehmet Keçeci'

# Sürüm Bilgisi (setuptools_scm kullanmıyorsanız sabit olarak tanımlayın)
# Gerçek sürümü modülden al (eğer mümkünse)
try:
    from kececilayout import __version__
    version = __version__
    release = __version__
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import __version__ from kececilayout: {e}")
    # Varsayılan değerler korunur
# version = '0.2.7'  # Geliştirme sürümü
# release = '0.2.7'  # Yayın sürümü

# Ana belge
master_doc = 'index'

# Sphinx Uzantıları
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',       # "Show Source" linki
    'sphinx.ext.napoleon',       # Google/NumPy docstring desteği
    'sphinx.ext.intersphinx',    # Dış belgelere link (ör: Python, NetworkX)
    'sphinx.ext.autosummary',    # Otomatik özet tabloları
    'sphinx_rtd_theme',          # Read the Docs teması
]

# Otomatik Özet (autosummary)
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'exclude-members': '__weakref__'
}

# Mock Imports — RTD'de kurulamayan modülleri taklit et
autodoc_mock_imports = [
    "igraph",
    "networkit",
    "rustworkx",
    "graphillion",
    "itertools",  # Bu aslında stdlib ama bazen sorun çıkarabilir
]

# Dış Belge Linkleri (intersphinx)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Hariç Tutulan Dosyalar
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/.ipynb_checkpoints',
    'README.md'  # HTML'e çevrilmemesi için
]

# HTML Ayarları
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'        # Opsiyonel: logo
html_favicon = '_static/favicon.ico'  # Opsiyonel: favicon
html_title = "KeçeciLayout Docs"

# HTML Tema Seçenekleri
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
}

# -- Extra: Eğer kececilayout içinde sürüm varsa, onu kullan (opsiyonel) --
# try:
#     from kececilayout import __version__
#     version = __version__
#     release = __version__
# except (ImportError, AttributeError):
#     pass
