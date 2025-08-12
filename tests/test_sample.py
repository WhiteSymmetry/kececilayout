# -*- coding: utf-8 -*-
"""
Advanced tests for the kececilayout library.

This test suite uses pytest to verify the functionality of layout calculations,
graph type compatibility, error handling, and drawing function routing.
"""

import numpy as np
import pytest
import sys
from unittest.mock import patch

# Import the module to be tested
# Assume the code is in a file named `kececilayout_lib.py` in the same directory
# or properly installed in the environment.
# Note: Before testing, it's recommended to clean up the provided code by
# removing duplicate functions (e.g., keep only one canonical `kececi_layout` function).
# We will test against the most feature-complete versions of the functions.
from kececilayout import (
    find_max_node_id,
    kececi_layout, # Assuming this is the main, multi-library compatible function
    to_networkx,
    draw_kececi,
    _draw_internal # Also test the internal router
)

# Pytest markers to skip tests if optional libraries are not installed
nx = pytest.importorskip("networkx")
ig = pytest.importorskip("igraph")
rx = pytest.importorskip("rustworkx")
nk = pytest.importorskip("networkit")
gg = pytest.importorskip("graphillion")
plt = pytest.importorskip("matplotlib.pyplot")


# --- Test Fixtures: Reusable Test Data ---

@pytest.fixture
def nx_graph_simple():
    """A simple, predictable NetworkX graph with 5 nodes."""
    return nx.path_graph(5)

@pytest.fixture
def ig_graph_simple():
    """A simple igraph graph, equivalent to the NetworkX path graph."""
    return ig.Graph.TupleList([(0, 1), (1, 2), (2, 3), (3, 4)], directed=False)

@pytest.fixture
def rx_graph_simple():
    """A simple Rustworkx graph."""
    g = rx.PyGraph()
    g.add_nodes_from(range(5))
    g.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None), (3, 4, None)])
    return g

@pytest.fixture
def nk_graph_simple():
    """A simple NetworKit graph."""
    g = nk.graph.Graph(5)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    g.addEdge(2, 3)
    g.addEdge(3, 4)
    return g

# --- Test Classes for Organization ---

class TestFindMaxNodeId:
    """Tests for the find_max_node_id helper function."""

    def test_empty_edges(self):
        """Should return 0 for an empty list of edges."""
        assert find_max_node_id([]) == 0

    def test_standard_edges(self):
        """Should find the max ID in a standard list."""
        edges = [(1, 2), (5, 3), (4, 0)]
        assert find_max_node_id(edges) == 5

    def test_with_malformed_data(self, capsys):
        """Should handle TypeError and return 0, printing a warning."""
        edges = [1, 2, 3] # Not a list of tuples
        assert find_max_node_id(edges) == 0
        captured = capsys.readouterr()
        assert "Warning: Edge format was unexpected" in captured.out

class TestKececiLayout:
    """Comprehensive tests for the main kececi_layout function."""

    def test_empty_graph(self, nx_graph_simple):
        """Should return an empty dict for an empty graph."""
        empty_graph = nx.Graph()
        assert kececi_layout(empty_graph) == {}

    def test_single_node_graph(self):
        """A single node should be positioned at the origin (0,0)."""
        g = nx.Graph()
        g.add_node("A")
        pos = kececi_layout(g)
        assert pos == {"A": (0.0, 0.0)}

    def test_invalid_primary_direction(self, nx_graph_simple):
        """Should raise ValueError for an invalid primary_direction."""
        with pytest.raises(ValueError, match="Invalid primary_direction"):
            kececi_layout(nx_graph_simple, primary_direction='diagonal')

    def test_invalid_secondary_start_vertical(self, nx_graph_simple):
        """Should raise ValueError for an invalid secondary_start in vertical mode."""
        with pytest.raises(ValueError, match="Invalid secondary_start for vertical"):
            kececi_layout(nx_graph_simple, primary_direction='top_down', secondary_start='up')

    def test_invalid_secondary_start_horizontal(self, nx_graph_simple):
        """Should raise ValueError for an invalid secondary_start in horizontal mode."""
        with pytest.raises(ValueError, match="Invalid secondary_start for horizontal"):
            kececi_layout(nx_graph_simple, primary_direction='left-to-right', secondary_start='left')

    def test_unsupported_graph_type(self):
        """Should raise TypeError for an unsupported object."""
        class UnknownGraph:
            pass
        with pytest.raises(TypeError, match="Unsupported graph type"):
            kececi_layout(UnknownGraph())

    @pytest.mark.parametrize("expanding, expected_x_coords", [
        (True,  [0.0, 1.0, -1.0, 2.0, -2.0]), # Expanding v4 style
        (False, [0.0, 1.0, -1.0, 1.0, -1.0])  # Parallel style
    ])
    def test_expanding_parameter(self, nx_graph_simple, expanding, expected_x_coords):
        """Test the effect of the 'expanding' parameter on coordinates."""
        pos = kececi_layout(nx_graph_simple,
                                primary_direction='top_down',
                                secondary_start='right',
                                expanding=expanding)
        
        assert len(pos) == 5
        # Check X coordinates (secondary axis)
        x_coords = [pos[i][0] for i in sorted(pos.keys())]
        np.testing.assert_allclose(x_coords, expected_x_coords)
        
        # Check Y coordinates (primary axis should be sequential)
        y_coords = [pos[i][1] for i in sorted(pos.keys())]
        np.testing.assert_allclose(y_coords, [0.0, -1.0, -2.0, -3.0, -4.0])

    def test_known_output_left_to_right(self):
        """Verify the exact coordinates for a known horizontal layout."""
        g = nx.path_graph(4) # Nodes 0, 1, 2, 3
        pos = kececi_layout(g,
                                primary_direction='left-to-right',
                                secondary_start='up',
                                expanding=True,
                                primary_spacing=2.0,
                                secondary_spacing=0.5)

        expected_pos = {
            0: (0.0,  0.0),   # x=primary, y=secondary
            1: (2.0,  0.5),   # y = 1 * ceil(1/2) * +1 * 0.5 = 0.5
            2: (4.0, -0.5),   # y = 1 * ceil(2/2) * -1 * 0.5 = -0.5
            3: (6.0,  1.0)    # y = 1 * ceil(3/2) * +1 * 0.5 = 1.0
        }

        assert pos.keys() == expected_pos.keys()
        for node in pos:
            np.testing.assert_allclose(pos[node], expected_pos[node], atol=1e-7)


class TestGraphCompatibility:
    """Ensures layout and conversion functions work with all supported libraries."""

    @pytest.mark.parametrize("graph_fixture", [
        "nx_graph_simple", "ig_graph_simple", "rx_graph_simple", "nk_graph_simple"
    ])
    def test_kececi_layout_on_all_types(self, graph_fixture, request):
        """Check that kececi_layout runs without error and returns a valid position dict."""
        graph = request.getfixturevalue(graph_fixture)
        pos = kececi_layout(graph)
        assert isinstance(pos, dict)
        assert len(pos) == 5
        # All values in the dict should be tuples of length 2 (x, y)
        assert all(isinstance(v, tuple) and len(v) == 2 for v in pos.values())

    @pytest.mark.parametrize("graph_fixture", [
        "ig_graph_simple", "rx_graph_simple", "nk_graph_simple"
    ])
    def test_to_networkx_conversion(self, graph_fixture, request):
        """Verify that conversion to NetworkX preserves graph structure."""
        original_graph = request.getfixturevalue(graph_fixture)
        nx_graph = to_networkx(original_graph)
        
        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.number_of_nodes() == 5
        assert nx_graph.number_of_edges() == 4
        # Check one edge to be sure
        assert nx_graph.has_edge(2, 3)


class TestDrawingFunctions:
    """
    Tests the user-facing drawing functions, primarily using mocking to avoid
    generating plots during automated tests.
    """

    @pytest.fixture(autouse=True)
    def close_plots(self):
        """Fixture to close any plots created during tests."""
        yield
        plt.close('all')

    def test_draw_kececi_invalid_style(self, nx_graph_simple):
        """Should raise ValueError for an unknown style."""
        with pytest.raises(ValueError, match="Invalid style"):
            draw_kececi(nx_graph_simple, style="nonexistent_style")

    def test_draw_kececi_3d_on_2d_axis(self, nx_graph_simple):
        """Should raise ValueError if style is '3d' but axis is 2D."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="requires an axis with 'projection=\"3d\"'"):
            draw_kececi(nx_graph_simple, style='3d', ax=ax)

    def test_draw_kececi_creates_axis(self, nx_graph_simple, mocker):
        """Should create a new figure and axis if none is provided."""
        mock_add_subplot = mocker.patch('matplotlib.figure.Figure.add_subplot')
        draw_kececi(nx_graph_simple)
        mock_add_subplot.assert_called_once()
    
        def test_draw_internal_routing(self, nx_graph_simple, mocker):
            """
            Verify that draw_kececi correctly calls _draw_internal with the
            right parameters by mocking the internal function.
            """
            # ÖNEMLİ: Fonksiyonu import etmek yerine, tam yolunu string olarak veriyoruz.
            # Python, test çalışırken bu yoldaki fonksiyonu bizim için izleyecek.
            mock_internal_draw = mocker.patch('kececilayout._draw_internal')
            
            # kececi_layout'u da mock'layalım ki sadece yönlendirmeyi test edelim.
            mocker.patch('kececilayout.kececi_layout', return_value={0:(0,0)})

            fig, ax = plt.subplots()
            
            # Genel (public) fonksiyon olan draw_kececi'yi çağırıyoruz.
            draw_kececi(nx_graph_simple, ax=ax, style='curved', 
                        expanding=False, node_size=500, primary_direction='bottom-up')

            # ŞİMDİ DOĞRULAMA:
            # _draw_internal fonksiyonumuz beklendiği gibi çağrıldı mı?
            mock_internal_draw.assert_called_once()
            
            # Çağrılırken verilen argümanları kontrol edelim.
            # kwargs, çağrının ikinci argümanıdır (args, kwargs).
            call_kwargs = mock_internal_draw.call_args[1]
            
            # Yerleşim (layout) parametreleri doğru aktarıldı mı?
            assert call_kwargs['expanding'] is False
            assert call_kwargs['primary_direction'] == 'bottom-up'
            
            # Çizim (drawing) parametreleri doğru aktarıldı mı?
            assert call_kwargs['node_size'] == 500

if __name__ == "__main__":
    pytest.main([__file__])  # Testleri direkt çalıştırırsa
    sys.exit(0)
