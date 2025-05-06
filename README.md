# KececiLayout

[![PyPI version](https://badge.fury.io/py/kececilayout.svg)](https://badge.fury.io/py/kececilayout) <!-- Opsiyonel: Eğer PyPI'daysa -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Opsiyonel: Lisans rozeti -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15313947.svg)](https://doi.org/10.5281/zenodo.15313947)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15314329.svg)](https://doi.org/10.5281/zenodo.15314329)

**Kececi Layout (Keçeci Yerleşimi)**: A deterministic graph layout algorithm designed for visualizing linear or sequential structures with a characteristic "zig-zag" or "serpentine" pattern.

*Python implementation of the Keçeci layout algorithm for graph visualization.*

---

## Description / Açıklama

This algorithm arranges nodes sequentially along a primary axis and offsets them alternately along a secondary axis. It's particularly useful for path graphs, chains, or showing progression.

*Bu algoritma, düğümleri birincil eksen boyunca sıralı olarak yerleştirir ve ikincil eksen boyunca dönüşümlü olarak kaydırır. Yol grafları, zincirler veya ilerlemeyi göstermek için özellikle kullanışlıdır.*

---

### English Description

**Keçeci Layout:**

A deterministic node placement algorithm used in graph visualization. In this layout, nodes are arranged sequentially along a defined primary axis. Each subsequent node is then alternately offset along a secondary, perpendicular axis, typically moving to one side of the primary axis and then the other. Often, the magnitude of this secondary offset increases as nodes progress along the primary axis, creating a characteristic "zig-zag" or "serpentine" pattern.

**Key Characteristics:**
*   **Linear Focus:** Particularly useful for visualizing linear or sequential structures, such as paths, chains, or ordered processes.
*   **Deterministic:** Produces the exact same layout for the same graph and parameters every time.
*   **Overlap Reduction:** Helps prevent node collisions by spreading nodes out away from the primary axis.
*   **Parametric:** Can be customized using parameters such as the primary direction (e.g., `top-down`), the starting side for the secondary offset (e.g., `start_right`), and the spacing along both axes (`primary_spacing`, `secondary_spacing`).

---

### Türkçe Tanımlama

**Keçeci Yerleşimi (Keçeci Layout):**

Graf görselleştirmede kullanılan deterministik bir düğüm yerleştirme algoritmasıdır. Bu yöntemde düğümler, belirlenen birincil (ana) eksen boyunca sıralı olarak yerleştirilir. Her bir sonraki düğüm, ana eksenin bir sağına bir soluna (veya bir üstüne bir altına) olmak üzere, ikincil eksen doğrultusunda dönüşümlü olarak kaydırılır. Genellikle, ana eksende ilerledikçe ikincil eksendeki kaydırma miktarı artar ve bu da karakteristik bir "zıgzag" veya "yılanvari" desen oluşturur.

**Temel Özellikleri:**
*   **Doğrusal Odak:** Özellikle yollar (paths), zincirler veya sıralı süreçler gibi doğrusal veya ardışık yapıları görselleştirmek için kullanışlıdır.
*   **Deterministik:** Aynı graf ve parametrelerle her zaman aynı sonucu üretir.
*   **Çakışmayı Azaltma:** Düğümleri ana eksenden uzağa yayarak çakışmaları önlemeye yardımcı olur.
*   **Parametrik:** Ana eksenin yönü (örn. `top-down`), ikincil kaydırmanın başlangıç yönü (örn. `start_right`) ve eksenler arası boşluklar (`primary_spacing`, `secondary_spacing`) gibi parametrelerle özelleştirilebilir.

---

## Installation / Kurulum

```bash
conda install bilgi::kececilayout -y

pip install kececilayout
```
https://anaconda.org/bilgi/kececilayout

https://pypi.org/project/KececiLayout/

https://github.com/WhiteSymmetry/kececilayout

https://zenodo.org/records/15313947

https://zenodo.org/records/15314329

---

## Usage / Kullanım

The layout function generally accepts a graph object and returns positions.

### Example with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt
import kececilayout as kl # Assuming the main function is imported like this

# Create a graph
G = nx.path_graph(10)

# Calculate layout positions using the generic function
# (Assuming kl.kececi_layout_v4 is the main/generic function)
pos = kl.kececi_layout_v4(G,
                           primary_spacing=1.0,
                           secondary_spacing=0.5,
                           primary_direction='top-down',
                           secondary_start='right')

# Draw the graph
plt.figure(figsize=(6, 8))
nx.draw(G, pos=pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
plt.title("Keçeci Layout with NetworkX")
plt.axis('equal') # Ensure aspect ratio is equal
plt.show()
```

```python
import matplotlib.pyplot as plt
import math
import networkx as nx
import kececilayout as kl

try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout_v4 is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top-down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === NetworkX Example ===
try:
    import networkx as nx
    print("\n--- NetworkX Example ---")

    # Generate graph (Path graph)
    G_nx = nx.path_graph(N_NODES)
    print(f"NetworkX graph generated: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_nx = kl.kececi_layout_v4(G_nx, **LAYOUT_PARAMS)
    # print("NetworkX positions:", pos_nx) # Debug print if needed

    # Plot
    plt.figure(figsize=(6, 8)) # Suitable figure size for vertical layout
    nx.draw(G_nx,               # NetworkX graph object
            pos=pos_nx,         # Positions calculated by Kececi Layout
            with_labels=True,   # Show node labels (indices)
            node_color='skyblue',# Node color
            node_size=700,      # Node size
            font_size=10,       # Label font size
            edge_color='gray')  # Edge color

    plt.title(f"NetworkX ({N_NODES} Nodes) with Keçeci Layout") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio for correct spacing perception
    # plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("NetworkX is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the NetworkX example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- NetworkX Example Finished ---")
```

### Example with iGraph

```python
import igraph as ig
import matplotlib.pyplot as plt
# Assuming a specific function for igraph exists or the generic one handles it
from kececilayout import kececi_layout_v4_igraph # Adjust import if needed

# Create a graph
G = ig.Graph.Ring(10, circular=False) # Path graph equivalent
for i in range(G.vcount()):
     G.vs[i]["name"] = f"N{i}"

# Calculate layout positions (returns a list of coords)
pos_list = kececi_layout_v4_igraph(G,
                                    primary_spacing=1.5,
                                    secondary_spacing=1.0,
                                    primary_direction='left-to-right',
                                    secondary_start='up')
layout = ig.Layout(coords=pos_list)

# Draw the graph
fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(
    G,
    target=ax,
    layout=layout,
    vertex_label=G.vs["name"],
    vertex_color="lightblue",
    vertex_size=30
)
ax.set_title("Keçeci Layout with iGraph")
ax.set_aspect('equal', adjustable='box')
plt.show()
```

```python
import matplotlib.pyplot as plt
import math
import igraph as ig
import kececilayout as kl


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout_v4 is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top-down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === igraph Example ===
try:
    import igraph as ig
    print("\n--- igraph Example ---")

    # Generate graph (Path graph using Ring(circular=False))
    G_ig = ig.Graph.Ring(N_NODES, directed=False, circular=False)
    print(f"igraph graph generated: {G_ig.vcount()} vertices, {G_ig.ecount()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_ig = kl.kececi_layout_v4(G_ig, **LAYOUT_PARAMS)
    # print("igraph positions (dict):", pos_ig) # Debug print if needed

    # Convert positions dict to list ordered by vertex index for ig.plot
    layout_list_ig = []
    plot_possible = True
    if pos_ig: # Check if dictionary is not empty
        try:
            # Generate list: [pos_ig[0], pos_ig[1], ..., pos_ig[N-1]]
            layout_list_ig = [pos_ig[i] for i in range(G_ig.vcount())]
            # print("igraph layout (list):", layout_list_ig) # Debug print if needed
        except KeyError as e:
             print(f"ERROR: Key {e} not found while creating position list for igraph.")
             print("The layout function might not have returned positions for all vertices.")
             plot_possible = False # Cannot plot if list is incomplete
    else:
        print("ERROR: Keçeci Layout returned empty positions for igraph.")
        plot_possible = False

    # Plot using igraph's plotting capabilities
    print("Plotting graph using igraph.plot...")
    fig, ax = plt.subplots(figsize=(6, 8)) # Generate matplotlib figure and axes

    if plot_possible:
        ig.plot(G_ig,
                target=ax,           # Draw on the matplotlib axes
                layout=layout_list_ig, # Use the ORDERED LIST of coordinates
                vertex_label=[str(i) for i in range(G_ig.vcount())], # Labels 0, 1,...
                vertex_color='lightgreen',
                vertex_size=30,      # Note: igraph vertex_size scale differs
                edge_color='gray')
    else:
         ax.text(0.5, 0.5, "Plotting failed:\nMissing or incomplete layout positions.",
                 ha='center', va='center', color='red', fontsize=12) # Error message on plot

    ax.set_title(f"igraph ({N_NODES} Nodes) with Keçeci Layout") # Plot title
    ax.set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
    # ax.grid(False) # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("python-igraph is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the igraph example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- igraph Example Finished ---")
```python

---

## Supported Backends / Desteklenen Kütüphaneler

The layout functions are designed to work with graph objects from the following libraries:

*   **NetworkX:** (`networkx.Graph`, `networkx.DiGraph`, etc.)
*   **igraph:** (`igraph.Graph`)
*   **Rustworkx:** (Requires appropriate conversion or adapter function)
*   **Networkit:** (Requires appropriate conversion or adapter function)
*   **Graphillion:** (Requires appropriate conversion or adapter function)

*Note: Direct support might vary. Check specific function documentation for compatibility details.*

---

## License / Lisans

This project is licensed under the MIT License. See the `LICENSE` file for details.

```

**Ek Notlar:**

*   **Rozetler (Badges):** Başlangıçta PyPI ve Lisans rozetleri ekledim (yorum satırı içinde). Eğer projeniz PyPI'da yayınlandıysa veya bir CI/CD süreci varsa, ilgili rozetleri eklemek iyi bir pratiktir.
*   **LICENSE Dosyası:** `LICENSE` bölümünde bir `LICENSE` dosyasına referans verdim. Projenizin kök dizininde MIT lisans metnini içeren bir `LICENSE` dosyası oluşturduğunuzdan emin olun.
*   **İçe Aktarma Yolları:** Örneklerde `import kececilayout as kl` veya `from kececilayout import kececi_layout_v4_igraph` gibi varsayımsal içe aktarma yolları kullandım. Kendi paket yapınıza göre bunları ayarlamanız gerekebilir.
*   **Fonksiyon Adları:** Örneklerde `kececi_layout_v4` ve `kececi_layout_v4_igraph` gibi fonksiyon adlarını kullandım. Gerçek fonksiyon adlarınız farklıysa bunları güncelleyin.
*   **Görselleştirme:** Örneklere `matplotlib.pyplot` kullanarak temel görselleştirme adımlarını ekledim, bu da kullanıcıların sonucu nasıl görebileceğini gösterir. Eksen oranlarını eşitlemek (`axis('equal')` veya `set_aspect('equal')`) layout'un doğru görünmesi için önemlidir.
```

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@misc{kececi_2025_15313947,
  author       = {Keçeci, Mehmet},
  title        = {kececilayout},
  month        = may,
  year         = 2025,
  publisher    = {PyPI, Anaconda, Github, Zenodo},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.15313947},
  url          = {https://doi.org/10.5281/zenodo.15313947},
}

@misc{kececi_2025_15314329,
  author       = {Keçeci, Mehmet},
  title        = {Keçeci Layout},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.15314329},
  url          = {https://doi.org/10.5281/zenodo.15314329},
}
```
### APA

```
Keçeci, M. (2025). kececilayout (0.2.0). PyPI, Anaconda, GitHub, Zenodo. https://doi.org/10.5281/zenodo.15313947

Keçeci, M. (2025). Keçeci Layout. https://doi.org/10.5281/zenodo.15314329
```

### Chicago

```
Keçeci, Mehmet. “Kececilayout”. PyPI, Anaconda, GitHub, Zenodo, 01 May 2025. https://doi.org/10.5281/zenodo.15313947.

Keçeci, Mehmet. “Keçeci Layout”, 01 May 2025. https://doi.org/10.5281/zenodo.15314329.
```
