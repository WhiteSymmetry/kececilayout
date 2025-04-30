# KececiLayout

[![PyPI version](https://badge.fury.io/py/kececilayout.svg)](https://badge.fury.io/py/kececilayout) <!-- Opsiyonel: Eğer PyPI'daysa -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Opsiyonel: Lisans rozeti -->

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
pip install kececilayout
```

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
