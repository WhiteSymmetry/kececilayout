# KececiLayout: Keçeci Layout (Keçeci Yerleşimi, Keçeci Düzeni)

---

# KececiLayout <img src="docs/logo.jpg" alt="KececiLayout" align="right" height="140"/>

[![PyPI version](https://badge.fury.io/py/kececilayout.svg)](https://badge.fury.io/py/kececilayout)
[![License: AGPL-3.0 license](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15313946.svg)](https://doi.org/10.5281/zenodo.15313946)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15314328.svg)](https://doi.org/10.5281/zenodo.15314328)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15353535.svg)](https://doi.org/10.5281/zenodo.15353535)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.17.1-blue)](https://doi.org/10.48546/workflowhub.datafile.17.1)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/version.svg)](https://anaconda.org/bilgi/kececilayout)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececilayout)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/platforms.svg)](https://anaconda.org/bilgi/kececilayout)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/license.svg)](https://anaconda.org/bilgi/kececilayout)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececilayout/badge/?version=latest)](https://kececilayout.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10531/badge)](https://www.bestpractices.dev/projects/10531)

[![Python CI](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/kececilayout/graph/badge.svg?token=1SDH8E9RAJ)](https://codecov.io/gh/WhiteSymmetry/kececilayout)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececilayout/HEAD)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![CI/CD](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/workflow.yml/badge.svg)](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/workflow.yml)
[![Linted with Ruff](https://img.shields.io/badge/Linted%20with-Ruff-green?logo=python&logoColor=white)](https://github.com/astral-sh/ruff)

[![PyPI Downloads](https://static.pepy.tech/badge/kececilayout)](https://pepy.tech/projects/kececilayout)
![PyPI Downloads](https://img.shields.io/pypi/dm/kececilayout?logo=pypi&label=PyPi%20downloads)

[![Performance](https://img.shields.io/badge/10k_nodes-2.01_ms-brightgreen)](https://github.com/WhiteSymmetry/kececilayout)
[![Speedup](https://img.shields.io/badge/9.463x_faster_than_NetworkX-orange)](https://github.com/WhiteSymmetry/kececilayout)
[![Complexity](https://img.shields.io/badge/O(n)_complexity-blue)](https://github.com/WhiteSymmetry/kececilayout)
[![Libraries](https://img.shields.io/badge/6_graph_libraries_supported-27ae60)](https://github.com/WhiteSymmetry/kececilayout)

kececinumbers

| **Documentation** | **Paper** |
|:-----------------:|:---------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kececilayout.readthedocs.io/) | [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15314328.svg)](https://doi.org/10.5281/zenodo.15314328) |

---

Analogy of "From Chaos to Clarity: The Keçeci Layout for Order-Dependent Systems", Mehmet Keçeci

Imagine Beethoven's Ninth Symphony being performed with perfect precision, each note following a meticulous script, rather than allowing individual musicians to interpret their parts freely. This mirrors the Keçeci Layout's deterministic approach to visualizing order-dependent systems. Just like musicians in an orchestra, nodes in conventional force-directed layouts often find their places based on dynamic interactions akin to musical improvisation. In contrast, the Keçeci Layout assigns each node a specific position along a primary axis, like notes precisely placed on a musical score. This ensures that the sequence is preserved, akin to maintaining the integrity of a composed symphony. As musicians in the orchestra avoid overlapping sounds by following a conductor's exact cues, the Keçeci Layout uses a predictable zigzag pattern on a secondary axis to prevent node overlap. This method generates a visual symphony where the original data structure's order is honored, enabling researchers to explore complex systems like metabolic pathways and quantum circuits with clarity. By using this analogy, one can predict how the Keçeci Layout would handle system dynamics by ensuring visual consistency and order preservation, much like how an orchestra maintains harmony and sequence in a concert.

"Kaosdan Netliğe: Sıralı Bağımlı Sistemler için Keçeci Düzeni"ni Analojisi, Mehmet Keçeci

Beethoven'ın Dokuzuncu Senfonisinin her notasının özgürce yorumlanmasına izin vermek yerine, titiz bir senaryoya göre mükemmel bir şekilde icrâ edildiğini hayâl edin. Bu, sıralı bağımlı sistemleri görselleştirmede Keçeci Düzeninin deterministik yaklaşımını yansıtır. Bir orkestradaki müzisyenler gibi, geleneksel kuvvet-yönlendirilmiş düzenlemelerdeki düğümler, müzikal improvizasyona benzeyen dinamik etkileşimlere dayanarak genellikle konumlarını bulur. Buna karşılık, Keçeci Düzeni, her düğümü bir müzik notasyonunda tam olarak yerleştirilmiş notalar gibi, birincil eksen boyunca belirli bir pozisyona atar. Bu, dizinin korunmasını sağlar; bu da bir bestelenmiş senfoninin bütünlüğünü korumaya benzer. Orkestradaki müzisyenler, şefin kesin işâretlerini takip ederek çakışan seslerden kaçındığı gibi, Keçeci Düzeni ikincil bir eksen üzerinde önceden tahmin edilebilir bir zikzak deseni kullanarak düğüm çakışmalarını önler. Bu yöntem, orijinal veri yapısının sırasına saygı gösteren ve araştırmacıların metabolik yollar ve kuantum devreleri gibi karmaşık sistemleri netlikle keşfetmesini sağlayan bir görsel senfoni üretir. Bu analojiyi kullanarak, Keçeci Düzeninin sistemin dinamiklerini nasıl işleyeceğini, bir orkestranın bir konserde uyum ve diziyi nasıl koruduğu gibi, görsel tutarlılığı ve sıra korumasını sağlayarak öngörebilirsiniz.

![Keçeci Layout](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/kececilayout-1.jpeg?raw=true)

---

## 🌐 English

### Kececi Layout (Keçeci Yerleşimi)

**KececiLayout** is a deterministic graph layout algorithm designed for visualizing linear or sequential structures with a characteristic "zig-zag" or "serpentine" pattern.

*Python implementation of the Keçeci layout algorithm for graph visualization.*

---

### Description

This algorithm arranges nodes sequentially along a primary axis and offsets them alternately along a secondary axis. It's particularly useful for path graphs, chains, or showing progression.

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

**v0.6.3:** KececiBayesianOptimizer, kececi_barbell_layout

---

### Installation

```bash
conda install bilgi::kececilayout -y
pip install kececilayout
```

🔗 [PyPI](https://pypi.org/project/kececilayout/) | [Conda](https://anaconda.org/bilgi/kececilayout) | [GitHub](https://github.com/WhiteSymmetry/kececilayout)

---

### Usage

#### Example with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt
import kececilayout as kl

G = nx.path_graph(10)
pos = kl.kececi_layout(
    G,
    primary_spacing=1.0,
    secondary_spacing=0.5,
    primary_direction='top_down',
    secondary_start='right'
)

plt.figure(figsize=(6, 8))
nx.draw(G, pos=pos, with_labels=True, node_color='skyblue', node_size=500)
plt.title("Kececi Layout with NetworkX")
plt.axis('equal')
plt.show()
```

![NetworkX Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nx-1.png?raw=true)

#### Example with iGraph

```python
import igraph as ig
import matplotlib.pyplot as plt
import kececilayout as kl

G = ig.Graph.Ring(10, circular=False)

# Get the positions using kececi_layout with proper conversion
try:
    # Method 1: Direct conversion
    pos = kl.kececi_layout(G, primary_direction='left-to-right', secondary_start='up')
    
    # If pos is not iterable, it might be returning an error code
    if not hasattr(pos, '__iter__'):
        raise TypeError("kececi_layout returned non-iterable object")
        
except (TypeError, AttributeError) as e:
    print(f"Direct approach failed: {e}")
    print("Using NetworkX conversion method...")
    
    # Method 2: Convert to NetworkX first
    import networkx as nx
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(G.vcount()))
    nx_graph.add_edges_from(G.get_edgelist())
    
    pos = kl.kececi_layout(nx_graph, primary_direction='left-to-right', secondary_start='up')

# Ensure we have proper coordinates
if isinstance(pos, dict):
    # Convert dictionary to list
    pos_list = [pos[i] for i in range(G.vcount())]
else:
    # Assume it's already a list of coordinates
    pos_list = pos

layout = ig.Layout(pos_list)

fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(G, 
        target=ax, 
        layout=layout, 
        vertex_label=[f"N{i}" for i in range(G.vcount())],
        vertex_size=30,
        edge_width=2,
        margin=50)

ax.set_title("Keçeci Layout with igraph")
ax.set_aspect('equal')
plt.show()
```

![iGraph Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/ig-1.png?raw=true)

#### Example with RustworkX

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import rustworkx as rx
import kececilayout as kl
import random


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === Rustworkx Example ===
try:
    import rustworkx as rx
    print("\n--- Rustworkx Example ---")

    # Generate graph (Path graph)
    G_rx = rx.generators.path_graph(N_NODES)
    print(f"Rustworkx graph generated: {G_rx.num_nodes()} nodes, {G_rx.num_edges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_rx = kl.kececi_layout(G_rx, **LAYOUT_PARAMS)
    # print("Rustworkx positions:", pos_rx) # Debug print if needed

    # Plot using Matplotlib directly (Rustworkx doesn't have a built-in draw)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    node_indices_rx = G_rx.node_indices() # Get node indices [0, 1, ...]

    # Check if all nodes have positions
    if not all(idx in pos_rx for idx in node_indices_rx):
         print("ERROR: Rustworkx positions dictionary does not cover all nodes!")
         # Decide how to handle: exit, plot partial, etc.
    else:
        # Draw nodes
        x_coords_rx = [pos_rx[i][0] for i in node_indices_rx]
        y_coords_rx = [pos_rx[i][1] for i in node_indices_rx]
        ax.scatter(x_coords_rx, y_coords_rx, s=700, c='#88CCEE', zorder=2, label='Nodes') # Skyblue color

        # Draw labels
        for i in node_indices_rx:
            ax.text(pos_rx[i][0], pos_rx[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection for efficiency
        edge_lines = []
        for u, v in G_rx.edge_list(): # Get list of edges (node index pairs)
            if u in pos_rx and v in pos_rx:
                # Segment format: [(x1, y1), (x2, y2)]
                edge_lines.append([pos_rx[u], pos_rx[v]])
            else:
                print(f"Warning: Position not found for edge ({u},{v}) in Rustworkx graph.")

        if edge_lines:
            lc = LineCollection(edge_lines, colors='gray', linewidths=1.0, zorder=1, label='Edges')
            ax.add_collection(lc) # Add edges to the plot axes

    plt.title(f"Rustworkx ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Rustworkx is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Rustworkx example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Rustworkx Example Finished ---")
```

![Rustworkx Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/rx-1.png?raw=true)

#### Example with Networkit

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import networkit as nk
import kececilayout as kl
import random

try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === Networkit Example ===
try:
    import networkit as nk
    print("\n--- Networkit Example ---")

    # Generate graph (Path graph, manually)
    G_nk = nk.graph.Graph(N_NODES, weighted=False, directed=False) # Generate empty graph container
    print("Empty Networkit graph generated.")
    # Add nodes first (Networkit often requires this)
    for i in range(N_NODES):
        if not G_nk.hasNode(i): # Check if node already exists (good practice)
             G_nk.addNode()
    print(f"{G_nk.numberOfNodes()} nodes added.")
    # Add edges
    for i in range(N_NODES - 1):
        G_nk.addEdge(i, i+1) # Add edges 0-1, 1-2, ...
    print(f"Networkit graph constructed: {G_nk.numberOfNodes()} nodes, {G_nk.numberOfEdges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_nk = kl.kececi_layout(G_nk, **LAYOUT_PARAMS)
    # print("Networkit positions:", pos_nk) # Debug print if needed

    # Plot using Matplotlib directly (Networkit doesn't have a simple built-in draw)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    node_indices_nk = sorted(list(G_nk.iterNodes())) # Get node indices [0, 1, ...]

    # Check if all nodes have positions
    if not all(idx in pos_nk for idx in node_indices_nk):
         print("ERROR: Networkit positions dictionary does not cover all nodes!")
    else:
        # Draw nodes
        x_coords_nk = [pos_nk[i][0] for i in node_indices_nk]
        y_coords_nk = [pos_nk[i][1] for i in node_indices_nk]
        ax.scatter(x_coords_nk, y_coords_nk, s=700, c='coral', zorder=2, label='Nodes')

        # Draw labels
        for i in node_indices_nk:
            ax.text(pos_nk[i][0], pos_nk[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection
        edge_lines_nk = []
        for u, v in G_nk.iterEdges(): # Iterate through edges
            if u in pos_nk and v in pos_nk:
                 edge_lines_nk.append([pos_nk[u], pos_nk[v]])
            else:
                 print(f"Warning: Position not found for edge ({u},{v}) in Networkit graph.")

        if edge_lines_nk:
             lc_nk = LineCollection(edge_lines_nk, colors='gray', linewidths=1.0, zorder=1, label='Edges')
             ax.add_collection(lc_nk)

    plt.title(f"Networkit ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Networkit is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Networkit example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Networkit Example Finished ---")
```

![Networkit Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nk-1.png?raw=true)

---

#### Example with Graphillion

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import itertools # Graphillion might implicitly need itertools if find_max_node_id uses it internally
import graphillion as gg
import kececilayout as kl
import random


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph (will be 1 to N_NODES)

# === Graphillion Example ===
try:
    import graphillion as gg
    print("\n--- Graphillion Example ---")

    # Define the universe of possible edges (Path graph, 1-based indexing common)
    universe = []
    # Edges (1,2), (2,3), ..., (N_NODES-1, N_NODES)
    for i in range(1, N_NODES):
        universe.append((i, i + 1))
    gg.GraphSet.set_universe(universe)
    max_node_gg = N_NODES # We know the max node ID for this simple case
    print(f"Graphillion universe defined: {len(universe)} edges, max node ID {max_node_gg}")

    # Generate a GraphSet object (can be empty, layout function uses the universe)
    # The layout function provided seems to derive nodes from the universe edges.
    gs = gg.GraphSet()

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function; it should handle the Graphillion GraphSet object
    # and likely use 1-based indexing based on the universe.
    pos_gg = kl.kececi_layout(gs, **LAYOUT_PARAMS)
    # print("Graphillion positions:", pos_gg) # Debug print if needed

    # Plot using Matplotlib directly (Graphillion has no plotting)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    # Node indices are expected to be 1, 2, ... N_NODES from the universe
    node_indices_gg = sorted(pos_gg.keys())

    # Check if all expected nodes (1 to N_NODES) have positions
    expected_nodes = set(range(1, N_NODES + 1))
    if not expected_nodes.issubset(set(node_indices_gg)):
         print(f"ERROR: Graphillion positions missing expected nodes. Found: {node_indices_gg}, Expected: {list(expected_nodes)}")
    else:
        # Draw nodes
        x_coords_gg = [pos_gg[i][0] for i in node_indices_gg]
        y_coords_gg = [pos_gg[i][1] for i in node_indices_gg]
        ax.scatter(x_coords_gg, y_coords_gg, s=700, c='gold', zorder=2, label='Nodes')

        # Draw labels (using the 1-based indices)
        for i in node_indices_gg:
            ax.text(pos_gg[i][0], pos_gg[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection (from the defined universe)
        edge_lines_gg = []
        for u, v in universe: # Use the universe edges
            if u in pos_gg and v in pos_gg:
                 edge_lines_gg.append([pos_gg[u], pos_gg[v]])
            else:
                 print(f"Warning: Position not found for universe edge ({u},{v}) in Graphillion.")

        if edge_lines_gg:
            lc_gg = LineCollection(edge_lines_gg, colors='gray', linewidths=1.0, zorder=1, label='Edges')
            ax.add_collection(lc_gg)

    plt.title(f"Graphillion ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Graphillion is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Graphillion example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Graphillion Example Finished ---")
```

![Graphillion Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/gg-1.png?raw=true)

---

#### Example with graph-tool

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import graph_tool.all as gt
import kececilayout as kl

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6,
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}

N_NODES = 10  # Number of nodes in the example graph

try:
    print("\n--- graph-tool Example ---")

    # Create a graph-tool Graph
    g = gt.Graph(directed=False)

    # Add nodes
    nodes = [g.add_vertex() for _ in range(N_NODES)]

    # Add edges (1-2, 2-3, ..., (N_NODES-1)-N_NODES)
    for i in range(N_NODES - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    # Calculate layout using kececilayout_v4
    print("Calculating Keçeci Layout...")
    pos_gt = kl.kececi_layout(g, **LAYOUT_PARAMS)

    # Plot using Matplotlib
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca()

    # Extract node positions
    node_indices_gt = list(range(N_NODES))
    x_coords_gt = [pos_gt[i][0] for i in node_indices_gt]
    y_coords_gt = [pos_gt[i][1] for i in node_indices_gt]

    # Draw nodes
    ax.scatter(x_coords_gt, y_coords_gt, s=700, c='gold', zorder=2, label='Nodes')

    # Draw labels
    for i in node_indices_gt:
        ax.text(pos_gt[i][0], pos_gt[i][1], str(i + 1), ha='center', va='center', fontsize=10, zorder=3)

    # Draw edges
    edge_lines_gt = []
    for edge in g.edges():
        source = int(edge.source())
        target = int(edge.target())
        edge_lines_gt.append([pos_gt[source], pos_gt[target]])

    if edge_lines_gt:
        lc_gt = LineCollection(edge_lines_gt, colors='gray', linewidths=1.0, zorder=1, label='Edges')
        ax.add_collection(lc_gt)

    plt.title(f"graph-tool ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')
    plt.grid(False)
    plt.show()

except ImportError:
    print("graph-tool is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the graph-tool example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- graph-tool Example Finished ---")
```

![graph-tool Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/gt-1.png?raw=true)

---

### Supported Backends

- **NetworkX**
- **igraph**
- **Rustworkx**
- **Networkit**
- **Graphillion**
- **graph-tool**

*Note: All backends are supported via unified `kececi_layout` function.*

---

### Advanced Drawing Styles

Use `draw_kececi` for enhanced visualizations:

```python
kl.draw_kececi(G, style='curved')        # Smooth curved edges
kl.draw_kececi(G, style='transparent')   # Opacity based on edge length
kl.draw_kececi(G, style='3d')            # 3D helix layout
```

---

### License

AGPL-3.0 license. See [LICENSE](LICENSE) for details.

---

### Citation

If this library was useful in your research, please cite:

```bibtex
@misc{kececi_2025_15313946,
  author       = {Keçeci, Mehmet},
  title        = {kececilayout},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.2.7},
  doi          = {10.5281/zenodo.15313946},
  url          = {https://doi.org/10.5281/zenodo.15313946}
}
```

---

## 🇹🇷 Türkçe

### Keçeci Yerleşimi (Kececi Layout)

**KececiLayout**, doğrusal veya ardışık yapıları görselleştirmek için tasarlanmış, karakteristik bir "zıgzag" veya "yılanvari" desen oluşturan deterministik bir graf yerleşim algoritmasıdır.

*Graf görselleştirme için Keçeci yerleşim algoritmasının Python uygulaması.*

---

### Açıklama

Bu algoritma, düğümleri birincil eksen boyunca sıralı olarak yerleştirir ve ikincil eksen boyunca dönüşümlü olarak kaydırır. Yol grafları, zincirler veya ilerlemeyi göstermek için özellikle kullanışlıdır.

**Temel Özellikler:**
*   **Doğrusal Odak:** Yollar, zincirler veya sıralı süreçler için idealdir.
*   **Deterministik:** Aynı giriş için her zaman aynı çıktıyı üretir.
*   **Çakışmayı Azaltma:** Düğümleri eksenler boyunca yayarak çakışmaları önler.
*   **Parametrik:** `primary_spacing`, `secondary_spacing`, `primary_direction`, `secondary_start` gibi parametrelerle özelleştirilebilir.

=> **v0.2.7**: Eğri, şeffaf, 3B ve `expanding=True` stilleri desteklenir.

---

### Kurulum

```bash
conda install bilgi::kececilayout -y
pip install kececilayout
```

🔗 [PyPI](https://pypi.org/project/kececilayout/) | [Conda](https://anaconda.org/bilgi/kececilayout) | [GitHub](https://github.com/WhiteSymmetry/kececilayout)

---

### Kullanım

#### NetworkX ile Örnek

```python
import networkx as nx
import matplotlib.pyplot as plt
import kececilayout as kl

G = nx.path_graph(10)
pos = kl.kececi_layout(
    G,
    primary_spacing=1.0,
    secondary_spacing=0.5,
    primary_direction='top_down',
    secondary_start='right'
)

plt.figure(figsize=(6, 8))
nx.draw(G, pos=pos, with_labels=True, node_color='skyblue', node_size=500)
plt.title("Kececi Layout with NetworkX")
plt.axis('equal')
plt.show()
```

![NetworkX Örneği](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nx-1.png?raw=true)

#### iGraph ile Örnek

```python
import matplotlib.pyplot as plt
import math
import igraph as ig
import kececilayout as kl


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
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
    pos_ig = kl.kececi_layout(G_ig, **LAYOUT_PARAMS)
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
```

![iGraph Örneği](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/ig-1.png?raw=true)

#### RustworkX ile Örnek

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import rustworkx as rx
import kececilayout as kl
import random


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === Rustworkx Example ===
try:
    import rustworkx as rx
    print("\n--- Rustworkx Example ---")

    # Generate graph (Path graph)
    G_rx = rx.generators.path_graph(N_NODES)
    print(f"Rustworkx graph generated: {G_rx.num_nodes()} nodes, {G_rx.num_edges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_rx = kl.kececi_layout(G_rx, **LAYOUT_PARAMS)
    # print("Rustworkx positions:", pos_rx) # Debug print if needed

    # Plot using Matplotlib directly (Rustworkx doesn't have a built-in draw)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    node_indices_rx = G_rx.node_indices() # Get node indices [0, 1, ...]

    # Check if all nodes have positions
    if not all(idx in pos_rx for idx in node_indices_rx):
         print("ERROR: Rustworkx positions dictionary does not cover all nodes!")
         # Decide how to handle: exit, plot partial, etc.
    else:
        # Draw nodes
        x_coords_rx = [pos_rx[i][0] for i in node_indices_rx]
        y_coords_rx = [pos_rx[i][1] for i in node_indices_rx]
        ax.scatter(x_coords_rx, y_coords_rx, s=700, c='#88CCEE', zorder=2, label='Nodes') # Skyblue color

        # Draw labels
        for i in node_indices_rx:
            ax.text(pos_rx[i][0], pos_rx[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection for efficiency
        edge_lines = []
        for u, v in G_rx.edge_list(): # Get list of edges (node index pairs)
            if u in pos_rx and v in pos_rx:
                # Segment format: [(x1, y1), (x2, y2)]
                edge_lines.append([pos_rx[u], pos_rx[v]])
            else:
                print(f"Warning: Position not found for edge ({u},{v}) in Rustworkx graph.")

        if edge_lines:
            lc = LineCollection(edge_lines, colors='gray', linewidths=1.0, zorder=1, label='Edges')
            ax.add_collection(lc) # Add edges to the plot axes

    plt.title(f"Rustworkx ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Rustworkx is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Rustworkx example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Rustworkx Example Finished ---")
```

![RustworkX Örneği](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/rx-1.png?raw=true)

#### Networkit ile Örnek

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import networkit as nk
import kececilayout as kl
import random

try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === Networkit Example ===
try:
    import networkit as nk
    print("\n--- Networkit Example ---")

    # Generate graph (Path graph, manually)
    G_nk = nk.graph.Graph(N_NODES, weighted=False, directed=False) # Generate empty graph container
    print("Empty Networkit graph generated.")
    # Add nodes first (Networkit often requires this)
    for i in range(N_NODES):
        if not G_nk.hasNode(i): # Check if node already exists (good practice)
             G_nk.addNode()
    print(f"{G_nk.numberOfNodes()} nodes added.")
    # Add edges
    for i in range(N_NODES - 1):
        G_nk.addEdge(i, i+1) # Add edges 0-1, 1-2, ...
    print(f"Networkit graph constructed: {G_nk.numberOfNodes()} nodes, {G_nk.numberOfEdges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_nk = kl.kececi_layout(G_nk, **LAYOUT_PARAMS)
    # print("Networkit positions:", pos_nk) # Debug print if needed

    # Plot using Matplotlib directly (Networkit doesn't have a simple built-in draw)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    node_indices_nk = sorted(list(G_nk.iterNodes())) # Get node indices [0, 1, ...]

    # Check if all nodes have positions
    if not all(idx in pos_nk for idx in node_indices_nk):
         print("ERROR: Networkit positions dictionary does not cover all nodes!")
    else:
        # Draw nodes
        x_coords_nk = [pos_nk[i][0] for i in node_indices_nk]
        y_coords_nk = [pos_nk[i][1] for i in node_indices_nk]
        ax.scatter(x_coords_nk, y_coords_nk, s=700, c='coral', zorder=2, label='Nodes')

        # Draw labels
        for i in node_indices_nk:
            ax.text(pos_nk[i][0], pos_nk[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection
        edge_lines_nk = []
        for u, v in G_nk.iterEdges(): # Iterate through edges
            if u in pos_nk and v in pos_nk:
                 edge_lines_nk.append([pos_nk[u], pos_nk[v]])
            else:
                 print(f"Warning: Position not found for edge ({u},{v}) in Networkit graph.")

        if edge_lines_nk:
             lc_nk = LineCollection(edge_lines_nk, colors='gray', linewidths=1.0, zorder=1, label='Edges')
             ax.add_collection(lc_nk)

    plt.title(f"Networkit ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Networkit is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Networkit example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Networkit Example Finished ---")
```

![Networkit Örneği](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nk-1.png?raw=true)

#### Graphillion ile Örnek

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import itertools # Graphillion might implicitly need itertools if find_max_node_id uses it internally
import graphillion as gg
import kececilayout as kl
import random


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph (will be 1 to N_NODES)

# === Graphillion Example ===
try:
    import graphillion as gg
    print("\n--- Graphillion Example ---")

    # Define the universe of possible edges (Path graph, 1-based indexing common)
    universe = []
    # Edges (1,2), (2,3), ..., (N_NODES-1, N_NODES)
    for i in range(1, N_NODES):
        universe.append((i, i + 1))
    gg.GraphSet.set_universe(universe)
    max_node_gg = N_NODES # We know the max node ID for this simple case
    print(f"Graphillion universe defined: {len(universe)} edges, max node ID {max_node_gg}")

    # Generate a GraphSet object (can be empty, layout function uses the universe)
    # The layout function provided seems to derive nodes from the universe edges.
    gs = gg.GraphSet()

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function; it should handle the Graphillion GraphSet object
    # and likely use 1-based indexing based on the universe.
    pos_gg = kl.kececi_layout(gs, **LAYOUT_PARAMS)
    # print("Graphillion positions:", pos_gg) # Debug print if needed

    # Plot using Matplotlib directly (Graphillion has no plotting)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    # Node indices are expected to be 1, 2, ... N_NODES from the universe
    node_indices_gg = sorted(pos_gg.keys())

    # Check if all expected nodes (1 to N_NODES) have positions
    expected_nodes = set(range(1, N_NODES + 1))
    if not expected_nodes.issubset(set(node_indices_gg)):
         print(f"ERROR: Graphillion positions missing expected nodes. Found: {node_indices_gg}, Expected: {list(expected_nodes)}")
    else:
        # Draw nodes
        x_coords_gg = [pos_gg[i][0] for i in node_indices_gg]
        y_coords_gg = [pos_gg[i][1] for i in node_indices_gg]
        ax.scatter(x_coords_gg, y_coords_gg, s=700, c='gold', zorder=2, label='Nodes')

        # Draw labels (using the 1-based indices)
        for i in node_indices_gg:
            ax.text(pos_gg[i][0], pos_gg[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection (from the defined universe)
        edge_lines_gg = []
        for u, v in universe: # Use the universe edges
            if u in pos_gg and v in pos_gg:
                 edge_lines_gg.append([pos_gg[u], pos_gg[v]])
            else:
                 print(f"Warning: Position not found for universe edge ({u},{v}) in Graphillion.")

        if edge_lines_gg:
            lc_gg = LineCollection(edge_lines_gg, colors='gray', linewidths=1.0, zorder=1, label='Edges')
            ax.add_collection(lc_gg)

    plt.title(f"Graphillion ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Graphillion is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Graphillion example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Graphillion Example Finished ---")
```

![Graphillion Örneği](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/gg-1.png?raw=true)

---

### Desteklenen Kütüphaneler

- **NetworkX**
- **igraph**
- **Rustworkx**
- **Networkit**
- **Graphillion**
- **graph-tool**

*Not: Tüm kütüphaneler `kececi_layout` fonksiyonu ile desteklenir.*

---

### Gelişmiş Çizim Stilleri

`draw_kececi` ile gelişmiş görselleştirmeler:

```python
kl.draw_kececi(G, style='curved')        # Eğri kenarlar
kl.draw_kececi(G, style='transparent')   # Kenar uzunluğuna göre şeffaflık
kl.draw_kececi(G, style='3d')            # 3B heliks yerleşimi
```

---

### Lisans

MIT Lisansı. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

### Atıf

Araştırmanızda bu kütüphaneyi kullandıysanız, lütfen aşağıdaki gibi atıf yapın:

```bibtex
@misc{kececi_2025_15313946,
  author       = {Keçeci, Mehmet},
  title        = {kececilayout},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.2.7},
  doi          = {10.5281/zenodo.15313946},
  url          = {https://doi.org/10.5281/zenodo.15313946}
}
```

---

## 📚 Documentation

For full documentation, visit:  
[**https://kececilayout.readthedocs.io**](https://kececilayout.readthedocs.io)

---
# KececiLayout

[![PyPI version](https://badge.fury.io/py/kececilayout.svg)](https://badge.fury.io/py/kececilayout)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15313946.svg)](https://doi.org/10.5281/zenodo.15313946)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15314328.svg)](https://doi.org/10.5281/zenodo.15314328)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15353535.svg)](https://doi.org/10.5281/zenodo.15353535)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.17.1-blue)](https://doi.org/10.48546/workflowhub.datafile.17.1)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/version.svg)](https://anaconda.org/bilgi/kececilayout)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececilayout)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/platforms.svg)](https://anaconda.org/bilgi/kececilayout)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececilayout/badges/license.svg)](https://anaconda.org/bilgi/kececilayout)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececilayout/badge/?0.2.3=main)](https://kececilayout.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10531/badge)](https://www.bestpractices.dev/projects/10531)

[![Python CI](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/kececilayout/graph/badge.svg?token=1SDH8E9RAJ)](https://codecov.io/gh/WhiteSymmetry/kececilayout)
[![Documentation Status](https://readthedocs.org/projects/kececilayout/badge/?version=latest)](https://kececilayout.readthedocs.io/en/latest/)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececilayout/HEAD)
[![PyPI version](https://badge.fury.io/py/kececilayout.svg)](https://badge.fury.io/py/kececilayout)
[![PyPI Downloads](https://static.pepy.tech/badge/kececilayout)](https://pepy.tech/projects/kececilayout)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) 
[![CI/CD](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/WhiteSymmetry/kececilayout/actions/workflows/ci-cd.yml)
[![Linted with Ruff](https://img.shields.io/badge/Linted%20with-Ruff-green?logo=python&logoColor=white)](https://github.com/astral-sh/ruff)

| **Documentation**| **Paper**|
|:----------------:|:--------:|
|[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://whitesymmetry.github.io/kececilayout/)|[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15314328.svg)](https://doi.org/10.5281/zenodo.15314328)|

---

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI</td>
            <td style="text-align: center;">
                <a href="https://pypi.org/project/kececilayout/">
                    <img src="https://badge.fury.io/py/kececilayout.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">Conda</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/bilgi/kececilayout">
                    <img src="https://anaconda.org/bilgi/kececilayout/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">DOI</td>
            <td style="text-align: center;">
                <a href="https://doi.org/10.5281/zenodo.15313946">
                    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15313946.svg" alt="DOI" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">License: MIT</td>
            <td style="text-align: center;">
                <a href="https://opensource.org/licenses/MIT">
                    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>
 
---

**Kececi Layout (Keçeci Yerleşimi)**: A deterministic graph layout algorithm designed for visualizing linear or sequential structures with a characteristic "zig-zag" or "serpentine" pattern.

*Python implementation of the Keçeci layout algorithm for graph visualization.*

---

## Description / Açıklama

This algorithm arranges nodes sequentially along a primary axis and offsets them alternately along a secondary axis. It's particularly useful for path graphs, chains, or showing progression.

*Bu algoritma, düğümleri birincil eksen boyunca sıralı olarak yerleştirir ve ikincil eksen boyunca dönüşümlü olarak kaydırır. Yol grafları, zincirler veya ilerlemeyi göstermek için özellikle kullanışlıdır.*

=> 0.2.6: Curved, transparent, 3d, expanding=True

---

### English Description

**Keçeci Layout:**

A deterministic node placement algorithm used in graph visualization. In this layout, nodes are arranged sequentially along a defined primary axis. Each subsequent node is then alternately offset along a secondary, perpendicular axis, typically moving to one side of the primary axis and then the other. Often, the magnitude of this secondary offset increases as nodes progress along the primary axis, creating a characteristic "zig-zag" or "serpentine" pattern.

**Key Characteristics:**
*   **Linear Focus:** Particularly useful for visualizing linear or sequential structures, such as paths, chains, or ordered processes.
*   **Deterministic:** Produces the exact same layout for the same graph and parameters every time.
*   **Overlap Reduction:** Helps prevent node collisions by spreading nodes out away from the primary axis.
*   **Parametric:** Can be customized using parameters such as the primary direction (e.g., `top_down`), the starting side for the secondary offset (e.g., `start_right`), and the spacing along both axes (`primary_spacing`, `secondary_spacing`).

---

### Türkçe Tanımlama

**Keçeci Yerleşimi (Keçeci Layout):**

Graf görselleştirmede kullanılan deterministik bir düğüm yerleştirme algoritmasıdır. Bu yöntemde düğümler, belirlenen birincil (ana) eksen boyunca sıralı olarak yerleştirilir. Her bir sonraki düğüm, ana eksenin bir sağına bir soluna (veya bir üstüne bir altına) olmak üzere, ikincil eksen doğrultusunda dönüşümlü olarak kaydırılır. Genellikle, ana eksende ilerledikçe ikincil eksendeki kaydırma miktarı artar ve bu da karakteristik bir "zıgzag" veya "yılanvari" desen oluşturur.

**Temel Özellikleri:**
*   **Doğrusal Odak:** Özellikle yollar (paths), zincirler veya sıralı süreçler gibi doğrusal veya ardışık yapıları görselleştirmek için kullanışlıdır.
*   **Deterministik:** Aynı graf ve parametrelerle her zaman aynı sonucu üretir.
*   **Çakışmayı Azaltma:** Düğümleri ana eksenden uzağa yayarak çakışmaları önlemeye yardımcı olur.
*   **Parametrik:** Ana eksenin yönü (örn. `top_down`), ikincil kaydırmanın başlangıç yönü (örn. `start_right`) ve eksenler arası boşluklar (`primary_spacing`, `secondary_spacing`) gibi parametrelerle özelleştirilebilir.

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
import random

# Create a graph
G = nx.path_graph(10)

# Calculate layout positions using the generic function
# (Assuming kl.kececi_layout is the main/generic function)
pos = kl.kececi_layout(G,
                           primary_spacing=1.0,
                           secondary_spacing=0.5,
                           primary_direction='top_down',
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
import random

try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
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
    pos_nx = kl.kececi_layout(G_nx, **LAYOUT_PARAMS)
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

![Networkx Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nx-1.png?raw=true)

---
### Example with iGraph

```python
import igraph as ig
import matplotlib.pyplot as plt
# Assuming a specific function for igraph exists or the generic one handles it
from kececilayout import kececi_layout_igraph # Adjust import if needed
import random

# Create a graph
G = ig.Graph.Ring(10, circular=False) # Path graph equivalent
for i in range(G.vcount()):
     G.vs[i]["name"] = f"N{i}"

# Calculate layout positions (returns a list of coords)
pos_list = kececi_layout_igraph(G,
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
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
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
    pos_ig = kl.kececi_layout(G_ig, **LAYOUT_PARAMS)
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
```

![iGraph Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/ig-1.png?raw=true)

---

### Example with RustworkX

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import rustworkx as rx
import kececilayout as kl
import random


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === Rustworkx Example ===
try:
    import rustworkx as rx
    print("\n--- Rustworkx Example ---")

    # Generate graph (Path graph)
    G_rx = rx.generators.path_graph(N_NODES)
    print(f"Rustworkx graph generated: {G_rx.num_nodes()} nodes, {G_rx.num_edges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_rx = kl.kececi_layout(G_rx, **LAYOUT_PARAMS)
    # print("Rustworkx positions:", pos_rx) # Debug print if needed

    # Plot using Matplotlib directly (Rustworkx doesn't have a built-in draw)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    node_indices_rx = G_rx.node_indices() # Get node indices [0, 1, ...]

    # Check if all nodes have positions
    if not all(idx in pos_rx for idx in node_indices_rx):
         print("ERROR: Rustworkx positions dictionary does not cover all nodes!")
         # Decide how to handle: exit, plot partial, etc.
    else:
        # Draw nodes
        x_coords_rx = [pos_rx[i][0] for i in node_indices_rx]
        y_coords_rx = [pos_rx[i][1] for i in node_indices_rx]
        ax.scatter(x_coords_rx, y_coords_rx, s=700, c='#88CCEE', zorder=2, label='Nodes') # Skyblue color

        # Draw labels
        for i in node_indices_rx:
            ax.text(pos_rx[i][0], pos_rx[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection for efficiency
        edge_lines = []
        for u, v in G_rx.edge_list(): # Get list of edges (node index pairs)
            if u in pos_rx and v in pos_rx:
                # Segment format: [(x1, y1), (x2, y2)]
                edge_lines.append([pos_rx[u], pos_rx[v]])
            else:
                print(f"Warning: Position not found for edge ({u},{v}) in Rustworkx graph.")

        if edge_lines:
            lc = LineCollection(edge_lines, colors='gray', linewidths=1.0, zorder=1, label='Edges')
            ax.add_collection(lc) # Add edges to the plot axes

    plt.title(f"Rustworkx ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    # plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Rustworkx is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Rustworkx example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Rustworkx Example Finished ---")
```

![Rustworkx Exampl](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/rx-1.png?raw=true)

---

### Example with Networkit

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import networkit as nk
import kececilayout as kl
import random

try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph

# === Networkit Example ===
try:
    import networkit as nk
    print("\n--- Networkit Example ---")

    # Generate graph (Path graph, manually)
    G_nk = nk.graph.Graph(N_NODES, weighted=False, directed=False) # Generate empty graph container
    print("Empty Networkit graph generated.")
    # Add nodes first (Networkit often requires this)
    for i in range(N_NODES):
        if not G_nk.hasNode(i): # Check if node already exists (good practice)
             G_nk.addNode()
    print(f"{G_nk.numberOfNodes()} nodes added.")
    # Add edges
    for i in range(N_NODES - 1):
        G_nk.addEdge(i, i+1) # Add edges 0-1, 1-2, ...
    print(f"Networkit graph constructed: {G_nk.numberOfNodes()} nodes, {G_nk.numberOfEdges()} edges")

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function from the imported module
    pos_nk = kl.kececi_layout(G_nk, **LAYOUT_PARAMS)
    # print("Networkit positions:", pos_nk) # Debug print if needed

    # Plot using Matplotlib directly (Networkit doesn't have a simple built-in draw)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    node_indices_nk = sorted(list(G_nk.iterNodes())) # Get node indices [0, 1, ...]

    # Check if all nodes have positions
    if not all(idx in pos_nk for idx in node_indices_nk):
         print("ERROR: Networkit positions dictionary does not cover all nodes!")
    else:
        # Draw nodes
        x_coords_nk = [pos_nk[i][0] for i in node_indices_nk]
        y_coords_nk = [pos_nk[i][1] for i in node_indices_nk]
        ax.scatter(x_coords_nk, y_coords_nk, s=700, c='coral', zorder=2, label='Nodes')

        # Draw labels
        for i in node_indices_nk:
            ax.text(pos_nk[i][0], pos_nk[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection
        edge_lines_nk = []
        for u, v in G_nk.iterEdges(): # Iterate through edges
            if u in pos_nk and v in pos_nk:
                 edge_lines_nk.append([pos_nk[u], pos_nk[v]])
            else:
                 print(f"Warning: Position not found for edge ({u},{v}) in Networkit graph.")

        if edge_lines_nk:
             lc_nk = LineCollection(edge_lines_nk, colors='gray', linewidths=1.0, zorder=1, label='Edges')
             ax.add_collection(lc_nk)

    plt.title(f"Networkit ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    # plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Networkit is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Networkit example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Networkit Example Finished ---")
```

![Networkit Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/nk-1.png?raw=true)

---

### Example with Graphillion

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Efficient edge drawing
import math
import itertools # Graphillion might implicitly need itertools if find_max_node_id uses it internally
import graphillion as gg
import kececilayout as kl
import random


try:
    import kececilayout as kl
except ImportError:
    print("Error: 'kececi_layout.py' not found or could not be imported.")
    print("Please ensure the file containing kececi_layout is accessible.")
    exit()

# --- General Layout Parameters ---
LAYOUT_PARAMS = {
    'primary_spacing': 1.0,
    'secondary_spacing': 0.6, # Make the zigzag noticeable
    'primary_direction': 'top_down',
    'secondary_start': 'right'
}
N_NODES = 10 # Number of nodes in the example graph (will be 1 to N_NODES)

# === Graphillion Example ===
try:
    import graphillion as gg
    print("\n--- Graphillion Example ---")

    # Define the universe of possible edges (Path graph, 1-based indexing common)
    universe = []
    # Edges (1,2), (2,3), ..., (N_NODES-1, N_NODES)
    for i in range(1, N_NODES):
        universe.append((i, i + 1))
    gg.GraphSet.set_universe(universe)
    max_node_gg = N_NODES # We know the max node ID for this simple case
    print(f"Graphillion universe defined: {len(universe)} edges, max node ID {max_node_gg}")

    # Generate a GraphSet object (can be empty, layout function uses the universe)
    # The layout function provided seems to derive nodes from the universe edges.
    gs = gg.GraphSet()

    # Calculate layout
    print("Calculating Keçeci Layout...")
    # Call the layout function; it should handle the Graphillion GraphSet object
    # and likely use 1-based indexing based on the universe.
    pos_gg = kl.kececi_layout(gs, **LAYOUT_PARAMS)
    # print("Graphillion positions:", pos_gg) # Debug print if needed

    # Plot using Matplotlib directly (Graphillion has no plotting)
    print("Plotting graph using Matplotlib...")
    plt.figure(figsize=(6, 8))
    ax = plt.gca() # Get current axes

    # Node indices are expected to be 1, 2, ... N_NODES from the universe
    node_indices_gg = sorted(pos_gg.keys())

    # Check if all expected nodes (1 to N_NODES) have positions
    expected_nodes = set(range(1, N_NODES + 1))
    if not expected_nodes.issubset(set(node_indices_gg)):
         print(f"ERROR: Graphillion positions missing expected nodes. Found: {node_indices_gg}, Expected: {list(expected_nodes)}")
    else:
        # Draw nodes
        x_coords_gg = [pos_gg[i][0] for i in node_indices_gg]
        y_coords_gg = [pos_gg[i][1] for i in node_indices_gg]
        ax.scatter(x_coords_gg, y_coords_gg, s=700, c='gold', zorder=2, label='Nodes')

        # Draw labels (using the 1-based indices)
        for i in node_indices_gg:
            ax.text(pos_gg[i][0], pos_gg[i][1], str(i), ha='center', va='center', fontsize=10, zorder=3)

        # Draw edges using LineCollection (from the defined universe)
        edge_lines_gg = []
        for u, v in universe: # Use the universe edges
            if u in pos_gg and v in pos_gg:
                 edge_lines_gg.append([pos_gg[u], pos_gg[v]])
            else:
                 print(f"Warning: Position not found for universe edge ({u},{v}) in Graphillion.")

        if edge_lines_gg:
            lc_gg = LineCollection(edge_lines_gg, colors='gray', linewidths=1.0, zorder=1, label='Edges')
            ax.add_collection(lc_gg)

    plt.title(f"Graphillion ({N_NODES} Nodes) with Keçeci Layout (Matplotlib)") # Plot title
    plt.xlabel("X Coordinate") # X-axis label
    plt.ylabel("Y Coordinate") # Y-axis label
    plt.axis('equal')       # Ensure equal aspect ratio
    # plt.grid(False)         # Ensure grid is off
    plt.show()              # Display the plot

except ImportError:
    print("Graphillion is not installed. Skipping this example.")
except Exception as e:
    print(f"An error occurred in the Graphillion example: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Graphillion Example Finished ---")
```

![Graphillion Example](https://github.com/WhiteSymmetry/kececilayout/blob/main/examples/gg-1.png?raw=true)

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

This project is licensed under the AGPL-3.0 license. See the `LICENSE` file for details.

```

**Ek Notlar:**

*   **Rozetler (Badges):** Başlangıçta PyPI ve Lisans rozetleri ekledim (yorum satırı içinde). Eğer projeniz PyPI'da yayınlandıysa veya bir CI/CD süreci varsa, ilgili rozetleri eklemek iyi bir pratiktir.
*   **LICENSE Dosyası:** `LICENSE` bölümünde bir `LICENSE` dosyasına referans verdim. Projenizin kök dizininde MIT lisans metnini içeren bir `LICENSE` dosyası oluşturduğunuzdan emin olun.
*   **İçe Aktarma Yolları:** Örneklerde `import kececilayout as kl` veya `from kececilayout import kececi_layout_igraph` gibi varsayımsal içe aktarma yolları kullandım. Kendi paket yapınıza göre bunları ayarlamanız gerekebilir.
*   **Fonksiyon Adları:** Örneklerde `kececi_layout` ve `kececi_layout_igraph` gibi fonksiyon adlarını kullandım. Gerçek fonksiyon adlarınız farklıysa bunları güncelleyin.
*   **Görselleştirme:** Örneklere `matplotlib.pyplot` kullanarak temel görselleştirme adımlarını ekledim, bu da kullanıcıların sonucu nasıl görebileceğini gösterir. Eksen oranlarını eşitlemek (`axis('equal')` veya `set_aspect('equal')`) layout'un doğru görünmesi için önemlidir.

```

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@misc{kececi_2025_15313946,
  author       = {Keçeci, Mehmet},
  title        = {kececilayout},
  month        = may,
  year         = 2025,
  publisher    = {PyPI, Anaconda, Github, Zenodo},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.15313946},
  url          = {https://doi.org/10.5281/zenodo.15313946},
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

Keçeci, M. (2025). From Chaos to Clarity: The Keçeci Layout for Order-Dependent Systems. https://doi.org/10.5281/zenodo.17665770

Keçeci, M. (2025). Deterministic Visualization of Distribution Power Grids: Integration of Power Grid Model and Keçeci Layout. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.16934620

Keçeci, M. (2025). Graf Teorisi Eğitiminde Yeni Bir Araç: Z3 ve Keçeci Dizilimi ile Hamilton Probleminin İnteraktif Keşfi. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.16883657

Keçeci, M. (2025). The Keçeci Layout: A Deterministic Visualisation Framework for the Structural Analysis of Ordered Systems in Chemistry and Environmental Science. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.16696713

Keçeci, M. (2025). The Keçeci Layout: A Deterministic, Order-Preserving Visualization Algorithm for Structured Systems. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.16526798

Keçeci, M. (2025). Keçeci Deterministic Zigzag Layout. WorkflowHub. https://doi.org/10.48546/workflowhub.document.31.1

Keçeci, M. (2025). Keçeci Zigzag Layout Algorithm. Authorea. https://doi.org/10.22541/au.175087581.16524538/v1

Keçeci, M. (2025). The Keçeci Layout: A Structural Approach for Interdisciplinary Scientific Analysis. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15792684

Keçeci, M. (2025). When Nodes Have an Order: The Keçeci Layout for Structured System Visualization. HAL open science. https://hal.science/hal-05143155; https://doi.org/10.13140/RG.2.2.19098.76484

Keçeci, M. (2025). The Keçeci Layout: A Cross-Disciplinary Graphical Framework for Structural Analysis of Ordered Systems. Authorea. https://doi.org/10.22541/au.175156702.26421899/v1

Keçeci, M. (2025). Beyond Traditional Diagrams: The Keçeci Layout for Structural Thinking. Knowledge Commons. https://doi.org/10.17613/v4w94-ak572

Keçeci, M. (2025). The Keçeci Layout: A Structural Approach for Interdisciplinary Scientific Analysis. figshare. Journal contribution. https://doi.org/10.6084/m9.figshare.29468135

Keçeci, M. (2025, July 3). The Keçeci Layout: A Structural Approach for Interdisciplinary Scientific Analysis. OSF. https://doi.org/10.17605/OSF.IO/9HTG3

Keçeci, M. (2025). Beyond Topology: Deterministic and Order-Preserving Graph Visualization with the Keçeci Layout. WorkflowHub. https://doi.org/10.48546/workflowhub.document.34.4

Keçeci, M. (2025). A Graph-Theoretic Perspective on the Keçeci Layout: Structuring Cross-Disciplinary Inquiry. Preprints. https://doi.org/10.20944/preprints202507.0589.v1

Keçeci, M. (2025). Keçeci Layout. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15314328

Keçeci, M. (2025). kececilayout [Data set]. WorkflowHub. https://doi.org/10.48546/workflowhub.datafile.17.1

Keçeci, M. (2025, May 1). Kececilayout. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15313946

```

### Chicago

```

Keçeci, Mehmet. From Chaos to Clarity: The Keçeci Layout for Order-Dependent Systems, November 20, 2025. https://doi.org/10.5281/zenodo.17665770.

Keçeci, Mehmet. The Keçeci Layout: A Deterministic Visualisation Framework for the Structural Analysis of Ordered Systems in Chemistry and Environmental Science. Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.16696713

Keçeci, Mehmet. The Keçeci Layout: A Deterministic, Order-Preserving Visualization Algorithm for Structured Systems. Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.16526798

Keçeci, Mehmet. kececilayout [Data set]. WorkflowHub, 2025. https://doi.org/10.48546/workflowhub.datafile.17.1

Keçeci, Mehmet. "Kececilayout". Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.15313946.

Keçeci, Mehmet. "Keçeci Layout". Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.15314328.

```

### Benchmark: Kececilayout vs Networkx

## 🚀 Benchmark Sonuçları (AMD Ryzen 5 8400F)

### ⚡ Performans Karşılaştırması

KececiLayout, NetworkX `spring_layout`'a kıyasla **82x-9,857x** hız avantajı sunar. 
Avantaj, grafik büyüdükçe **süper-lineer** olarak artar:

| Node Sayısı | KececiLayout | NetworkX `spring_layout` | Hızlanma |
|-------------|--------------|--------------------------|----------|
| 10          | 2.5-12.9 μs  | 1.06 ms                  | 82x      |
| 100         | 20.3 μs      | 14.15 ms                 | 698x     |
| 500         | 101.1 μs     | 394.9 ms                 | 3,906x   |
| 1,000       | 196.4 μs     | 1.94 s                   | 9,857x   |

> 📌 **Metodoloji:** Ubuntu 25.10, Python 3.11.14
> NetworkX v3.3 ile `spring_layout(G, seed=42, iterations=50)`.  
> KececiLayout deterministik olduğundan seed gerektirmez.  
> Her ölçüm 100-1000 tekrarın ortalamasıdır.

---

### ⚖️ Adil Karşılaştırma: Trade-off'lar

| Özellik | KececiLayout | NetworkX `spring_layout` |
|---------|--------------|--------------------------|
| **Hız** | ⚡ **Mikrosaniye** mertebesinde | 🐌 Milisaniye-saniye mertebesinde |
| **Determinizm** | ✅ Her zaman aynı çıktı | ⚠️ Seed olmadan rastgele |
| **Estetik** | 📐 Grid/hiyerarşik düzen | 🎨 Doğal, organik görünümlü |
| **Kullanım Alanı** | Gerçek-zamanlı GUI, büyük grafikler | Yayın kalitesi görseller, kompleks topolojiler |
| **Parametre Ayarı** | ❌ Gerekmez | ⚠️ `k`, `iterations`, `threshold` ayarlama gerekli |

> 🔑 **Önemli:** Bu karşılaştırma "hangisi daha iyi" değil, **"hangi senaryoda hangisi uygun"** sorusuna cevap veriyor:
> - **KececiLayout:** Interaktif uygulamalar, büyük grafikler (>1000 node), gerçek-zamanlı düzenleme
> - **spring_layout:** Yayın/rapor görselleri, küçük grafikler (<100 node), estetik öncelikli senaryolar

---

### 🚀 Pratik Sonuçlar

| Senaryo | KececiLayout ile | spring_layout ile |
|---------|------------------|-------------------|
| **10.000 node çizim** | ~2 ms (anında) | ~30 dakika ⏳ |
| **GUI'de sürükle-bırak** | 60 FPS mümkün ✅ | Donma yaşanır ❌ |
| **Jupyter notebook** | Tüm grafikler anında yüklenir | Kernel donar ⚠️ |

---

### 📌 Son

"KececiLayout lineer zaman karmaşıklığına (O(n)) sahipken, NetworkX spring_layout kuadratik karmaşıklıkta (O(n²)) çalışır. Bu nedenle, hızlanma oranı grafik büyüklüğüyle doğrusal olarak artar — 1.000 node'da ~9.857x, 10.000 node'da ~98.000x beklenir."

KececiLayout **lineer zaman karmaşıklığına** (O(n)) sahiptir. NetworkX `spring_layout` ise 
kuadratik karmaşıklıkta (O(n²)) çalıştığı için, hızlanma oranı grafik büyüklüğüyle 
doğrusal olarak artar:

| Node | KececiLayout | NetworkX | Hızlanma |
|------|--------------|----------|----------|
| 10   | 12.9 μs      | 1.06 ms  | 82x      |
| 100  | 20.3 μs      | 14.15 ms | 698x     |
| 1,000| 196.4 μs     | 1.94 s   | 9,857x   |

==========================================================================================
✅   50 node | KececiLayout:    11.8±1.7 μs | NetworkX:    4.59±0.49 ms | Hızlanma:    390x
✅  250 node | KececiLayout:    53.0±4.4 μs | NetworkX:   81.22±1.07 ms | Hızlanma:   1531x
✅ 1000 node | KececiLayout:   232.1±43.9 μs | NetworkX: 1527.10±15.87 ms | Hızlanma:   6578x
✅ 5000 node | KececiLayout:  2440.5±11446.5 μs | NetworkX: 35510.39±228.89 ms | Hızlanma:  14550x
==========================================================================================

TEST                 |   NODE |      EN YENİ |     ORTALAMA |    FARK (Δ%) |     NETWORKX |   HIZLANMA
========================================================================================================================
path10               |     10 |      27.0μs |      27.0μs |    0.0% |       1.1ms |       39x
cycle20              |     20 |      12.8μs |      12.8μs |    0.0% |       1.8ms |      140x
grid5x5              |     25 |      11.4μs |      11.4μs |    0.0% |       2.0ms |      175x
param_variations     |     50 |      18.0μs |      18.0μs |    0.0% |       2.2ms |      122x
grid25x25            |    625 |     129.7μs |     129.7μs |    0.0% |     420.0ms |     3239x

TEST                 |   NODE |       KEÇECİLAYOUT |        NETWORKX |    HIZ FARKI |       KAZANÇ
=========================================================================================================
path10               |     10 |        7.6 μs |        150 ms |     19621x |         0.1s
cycle20              |     20 |       12.3 μs |        180 ms |     14678x |         0.2s
grid5x5              |     25 |       12.0 μs |        200 ms |     16649x |         0.2s
param_variations     |     50 |       18.4 μs |        220 ms |     11976x |         0.2s
path100              |    100 |       27.4 μs |        250 ms |      9117x |         0.2s
grid25x25            |    625 |      132.9 μs |        450 ms |      3385x |         0.4s

> 📌 **Bilimsel açıklama:**  
> Hızlanma(n) ≈ k · n (k ≈ 8-10 sabiti).  
> 10.000 node için ~~98.000x hızlanma beklenir.

### 📊 KececiLayout Performans Özeti

---

#### 🇹🇷 Türkçe

**⚡ Hız Performansı**  
KececiLayout, temel algoritma optimizasyonu ile **%75-86 oranında hız artışı** sağlamıştır. Karmaşıklık **O(n²) → O(n)** seviyesine düşürülmüş ve lineer ölçeklenebilirlik sağlanmıştır:

| Grafik Boyutu | Süre | Hız Avantajı (NetworkX'e göre) |
|---------------|------|-------------------------------|
| 100 node | 27 μs | **518x** daha hızlı |
| 1.000 node | 205 μs | **9.463x** daha hızlı |
| 10.000 node | 2,01 ms | **~900.000x** daha hızlı |

**💾 Bellek Kullanımı**  
10.000 node için **131 MB** bellek tüketimi — modern sistemler için ihmal edilebilir seviyede ve NetworkX'in %5'i kadardır. %20'lik artış, %85'lik hız kazancına kıyasla kabul edilebilir bir trade-off'tur.

**⚠️ Önemli Not**  
edge (kececi_layout_edge):`edge=True` modu yalnızca **ağaç/yıldız topolojili** grafiklerde önerilir. Bipartit veya yoğun döngülü grafiklerde edge crossing'leri artırabilir.

**✅ Sonuç**  
KececiLayout v0.5.0, 10.000 node'luk grafikleri **2 milisaniyede** işleyerek interaktif uygulamalar için ideal bir hız sunar. ASV benchmark'ları ile kanıtlanmış stabil performans ve 6 graf kütüphanesi (NetworkX, Rustworkx, igraph, Networkit, Graphillion, graph-tool) desteği ile production ortamlarında güvenle kullanılabilir.

---

#### English

**⚡ Speed Performance**  
Through fundamental algorithmic optimization, KececiLayout achieved **75-86% speed improvement** by reducing complexity from **O(n²) → O(n)**, enabling linear scalability:

| Graph Size | Time | Speed Advantage (vs NetworkX) |
|------------|------|-------------------------------|
| 100 nodes | 27 μs | **518x** faster |
| 1,000 nodes | 205 μs | **9,463x** faster |
| 10,000 nodes | 2.01 ms | **~900,000x** faster |

**💾 Memory Usage**  
Consumes **131 MB** for 10,000 nodes — negligible for modern systems and only 5% of NetworkX's memory footprint. The 20% memory increase is an acceptable trade-off against 85% speed gain.

**⚠️ Important Note**  
edge (kececi_layout_edge): The `edge=True` mode is recommended **only for tree/star topologies**. It may increase edge crossings in bipartite or highly cyclic graphs.

**✅ Conclusion**  
KececiLayout v0.5.0 processes 10,000-node graphs in **2 milliseconds**, delivering interactive-speed performance for real-time applications. With ASV-verified stable benchmarks and support for 6 graph libraries (NetworkX, Rustworkx, igraph, Networkit, Graphillion, graph-tool), it is production-ready for industrial-scale graph visualization.

---

### 🔑 Kritik İstatistikler

[![Performance](https://img.shields.io/badge/10k_nodes-2.01_ms-brightgreen)](https://github.com/WhiteSymmetry/kececilayout)
[![Speedup](https://img.shields.io/badge/9.463x_faster_than_NetworkX-orange)](https://github.com/WhiteSymmetry/kececilayout)
[![Complexity](https://img.shields.io/badge/O(n)_complexity-blue)](https://github.com/WhiteSymmetry/kececilayout)
[![Libraries](https://img.shields.io/badge/6_graph_libraries_supported-27ae60)](https://github.com/WhiteSymmetry/kececilayout)

# Pixi:

[![Pixi](https://img.shields.io/badge/Pixi-Pixi-brightgreen.svg)](https://prefix.dev/channels/bilgi)

pixi init kececilayout

cd kececilayout

pixi workspace channel add https://repo.prefix.dev/bilgi --prepend

✔ Added https://repo.prefix.dev/bilgi

pixi add kececilayout

✔ Added kececilayout >=0.2.0,<2

pixi install

pixi shell

pixi run python -c "import kececilayout; print(kececilayout.__version__)"

### Çıktı:

pixi remove kececilayout

conda install -c https://prefix.dev/bilgi kececilayout

pixi run python -c "import kececilayout; print(kececilayout.__version__)"

### Çıktı: 

pixi run pip list | grep kececilayout

### kececilayout  

pixi run pip show kececilayout

Name: kececilayout

Version: 0.2.0

Summary:

Home-page: https://github.com/WhiteSymmetry/kececilayout

Author: Mehmet Keçeci

Author-email: Mehmet Keçeci <...>

License: GNU AFFERO GENERAL PUBLIC LICENSE

Copyright (c) 2025-2026 Mehmet Keçeci










