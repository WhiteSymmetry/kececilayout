```markdown
# KececiLayout

Keçeci Layout (Kececi Layout, Keçeci Yerleşimi): A deterministic node placement algorithm used in graph visualization. In this layout, nodes are arranged sequentially along a defined primary axis. Each subsequent node is then alternately offset along a secondary, perpendicular axis, typically moving to one side of the primary axis and then the other. Often, the magnitude of this secondary offset increases as nodes progress along the primary axis, creating a characteristic "zig-zag" or "serpentine" pattern.

Python için gelişmiş çizim algoritması. Graflar üzerinde özel olarak sıralı yerleşim yapar.

"Keçeci Layout" için Türkçe ve İngilizce tanımlamalar:

---

# Türkçe Tanımlama

## Keçeci Yerleşimi (Keçeci Layout):

Graf görselleştirmede kullanılan deterministik bir düğüm yerleştirme algoritmasıdır. Bu yöntemde düğümler, belirlenen birincil (ana) eksen boyunca sıralı olarak yerleştirilir. Her bir sonraki düğüm, ana eksenin bir sağına bir soluna (veya bir üstüne bir altına) olmak üzere, ikincil eksen doğrultusunda dönüşümlü olarak kaydırılır. Genellikle, ana eksende ilerledikçe ikincil eksendeki kaydırma miktarı artar ve bu da karakteristik bir "zıgzag" veya "yılanvari" desen oluşturur.

### Temel Özellikleri:
*   **Doğrusal Odak:** Özellikle yollar (paths), zincirler veya sıralı süreçler gibi doğrusal veya ardışık yapıları görselleştirmek için kullanışlıdır.
*   **Deterministik:** Aynı graf ve parametrelerle her zaman aynı sonucu üretir.
*   **Çakışmayı Azaltma:** Düğümleri ana eksenden uzağa yayarak çakışmaları önlemeye yardımcı olur.
*   **Parametrik:** Ana eksenin yönü (örn. yukarıdan aşağı), ikincil kaydırmanın başlangıç yönü (örn. sağdan başla) ve eksenler arası boşluklar gibi parametrelerle özelleştirilebilir.

---

# English Description

##Keçeci Layout:

A deterministic node placement algorithm used in graph visualization. In this layout, nodes are arranged sequentially along a defined primary axis. Each subsequent node is then alternately offset along a secondary, perpendicular axis, typically moving to one side of the primary axis and then the other. Often, the magnitude of this secondary offset increases as nodes progress along the primary axis, creating a characteristic "zig-zag" or "serpentine" pattern.

###Key Characteristics:
*   **Linear Focus:** Particularly useful for visualizing linear or sequential structures, such as paths, chains, or ordered processes.
*   **Deterministic:** Produces the exact same layout for the same graph and parameters every time.
*   **Overlap Reduction:** Helps prevent node collisions by spreading nodes out away from the primary axis.
*   **Parametric:** Can be customized using parameters such as the primary direction (e.g., top-down), the starting side for the secondary offset (e.g., start right), and the spacing along both axes.

---

## Kurulum

```bash
pip install kececilayout
```

## Örnek Kullanım

```python
import networkx as nx
import kececilayout as kl

G = nx.path_graph(10)
pos = kl.kececi_layout_v4(G)
nx.draw(G, pos=pos, with_labels=True)
```

Desteklenen arka uçlar:
- NetworkX
- Rustworkx
- iGraph
- Networkit
- Graphillion
```



### Lisans (Türkçe) / License (English)

```
This project is licensed under the MIT License.
```
