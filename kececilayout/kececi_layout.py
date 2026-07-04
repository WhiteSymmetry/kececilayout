# kececifractals.py
"""
Keçeci Fractals: Keçeci Fraktalları (Keçeci Circle Fractal (KCF): Keçeci Dairesel Fraktalı (KDF), Keçeci-style circle fractal)
This module provides three primary functionalities for generating Keçeci Fractals:
1.  kececifractals_circle(): Generates general-purpose, aesthetic, and randomly
    colored circular fractals.
2.  visualize_qec_fractal(): Generates fractals customized for modeling the
    concept of Quantum Error Correction (QEC) codes.
3.  kececifractals_3d(): Generates 3D versions of Keçeci fractals.

pip install -U kececilayout matplotlib networkx numpy PyOpenGL pyopencl vulkan

* 0.2.5: GPU/OpenCL/OpenGL/Vulkan/Auto support
* 0.2.4: GPU/OpenCL support

"""

import ctypes
import math
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d
import networkx as nx  # STRATUM MODEL VISUALIZATION
import numpy as np
from OpenGL.EGL import * # pip install -U PyOpenGL
from OpenGL import GL
from OpenGL.GL import shaders
import os
# Linux için gerekirse bunları açabilirsin
# os.environ["RUSTICL_ENABLE"] = "radeonsi"
# os.environ["OCL_ICD_VENDORS"] = "/etc/OpenCL/vendors"
import pyopencl as cl # pip install -U pyopencl
import random
import subprocess
import sys
import vulkan as vk # pip install -U vulkan
from vulkan._vulkan import ffi
import tempfile
import warnings
from typing import Callable, List, Optional, Tuple, Union


# Import kececilayout if available, otherwise use a fallback
try:
    import kececilayout as kl  # STRATUM MODEL VISUALIZATION
except ImportError:
    # Fallback layout function if kececilayout is not available
    class kl:
        @staticmethod
        def kececi_layout(
            G, primary_direction="top_down", primary_spacing=1.5, secondary_spacing=1.0
        ):
            pos = {}
            for i, node in enumerate(G.nodes()):
                if primary_direction == "top_down":
                    pos[node] = (i * secondary_spacing, -i * primary_spacing)
                else:
                    pos[node] = (i * primary_spacing, i * secondary_spacing)
            return pos

def select_opencl_platform(prefer_rusticl=True):
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("Hiçbir OpenCL platformu bulunamadı.")
    if prefer_rusticl:
        for plat in platforms:
            if "rusticl" in plat.name.lower():
                return plat
    return platforms[0]

HIGH_CONTRAST_COLORS = [
    (1.0, 1.0, 1.0),   # Camgöbeği
    (1.0, 0.0, 1.0),   # Magenta
    (1.0, 1.0, 0.0),   # Sarı
    (0.0, 1.0, 0.0),   # Yeşil
    (1.0, 0.0, 0.0),   # Kırmızı
    (0.0, 0.0, 1.0),   # Mavi
    (1.0, 0.5, 0.0),   # Turuncu
]

class KececiFractalError(Exception):
    """Keçeci Fractals için temel exception."""

    pass


class FractalParameterError(KececiFractalError):
    """Fraktal parametre hatası."""

    pass


class ColorParseError(KececiFractalError):
    """Renk parse hatası."""

    pass


class ThreeDNotSupportedError(KececiFractalError):
    """3D desteklenmiyor hatası."""

    pass


class InvalidAxisError(KececiFractalError):
    """Geçersiz eksen hatası."""

    pass


# --- GENERAL HELPER FUNCTIONS ---
def random_soft_color():
    """Generates a random soft RGB color tuple."""
    return tuple(random.uniform(0.4, 0.95) for _ in range(3))


def _parse_color(
    color_input: Union[str, Tuple[float, float, float], None],
) -> Optional[Tuple[float, float, float]]:
    """
    Parses color input which can be:
    - None
    - RGB tuple (0-1 range)
    - Hex string like '#RRGGBB'
    - Named color like 'red', 'blue', etc.

    Returns RGB tuple in 0-1 range or None.
    """
    if color_input is None:
        return None

    # If already a tuple, assume it's correct format
    if isinstance(color_input, tuple):
        if len(color_input) == 3:
            return color_input
        elif len(color_input) == 4:
            return color_input[:3]  # Drop alpha if present

    # Try to parse as string
    if isinstance(color_input, str):
        try:
            # First try matplotlib's color conversion
            rgb = to_rgb(color_input)
            return rgb
        except (ValueError, AttributeError):
            # Try hex parsing manually
            if color_input.startswith("#"):
                try:
                    # Remove # and parse
                    hex_color = color_input.lstrip("#")
                    if len(hex_color) == 3:
                        # Expand shorthand #RGB to #RRGGBB
                        hex_color = "".join([c * 2 for c in hex_color])
                    elif len(hex_color) != 6:
                        raise ValueError(f"Invalid hex color: {color_input}")

                    # Convert to RGB 0-255
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    return (r, g, b)
                except:
                    pass

    # If we get here, return random color as fallback
    print(
        f"Warning: Could not parse color '{color_input}'. Using random color.",
        file=sys.stderr,
    )
    return random_soft_color()


def _draw_circle_patch(ax, center, radius, face_color, edge_color="black", lw=0.5):
    """
    A robust helper function that adds a circle patch to the Matplotlib axes,
    using facecolor and edgecolor to avoid the UserWarning.
    """
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=lw,
            fill=True,
        )
    )


# ==============================================================================
# PART 1: GENERAL-PURPOSE KEÇECİ FRACTALS
# ==============================================================================
def draw_sphere(
    ax,
    center,
    radius,
    color,
    alpha=0.8,
    resolution_u=20,
    resolution_v=12,
    edgecolor='k',
    linewidth=0.2,
    shade=True
):
    """
    Draw a 3D sphere using plot_surface.
    
    Backward-compatible with previous versions.
    Supports customizable resolution and styling.
    
    Parameters:
        ax: matplotlib 3D axis
        center: (x, y, z) tuple or array-like
        radius: float
        color: face color
        alpha: transparency (default: 0.8)
        resolution_u: longitudinal resolution (default: 20 → matches old behavior)
        resolution_v: latitudinal resolution (default: 12 → matches old behavior)
        edgecolor: color of mesh lines (default: 'k' for visible edges)
        linewidth: width of mesh lines (default: 0.2)
        shade: enable shading (default: True)
    """
    u = np.linspace(0, 2 * np.pi, resolution_u)
    v = np.linspace(0, np.pi, resolution_v)
    u, v = np.meshgrid(u, v)

    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)

    ax.plot_surface(
        x, y, z,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        shade=shade,
        antialiased=True
    )

def get_icosahedron_vertices():
    """Return 12 normalized vertices of an icosahedron for even 3D distribution."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

"""
def draw_3d_sphere(
    ax,
    center: Tuple[float, float, float],
    radius: float,
    color: Tuple[float, float, float],
    alpha: float = 1.0,
):
    # 3D eksen üzerine küre çizer.

    if not HAS_3D:
        return

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        edgecolor="none",
        antialiased=True,
        shade=True,
        linewidth=0.5,
    )
"""
def draw_3d_sphere(ax, center=(0,0,0), radius=1.0, color='cyan', alpha=0.3):
    """🌀 3D Küre"""
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:12j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    
    if isinstance(color, (tuple, list)):
        color = color
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none",
                   antialiased=True, shade=True, linewidth=0.5)

def draw_kececi_spiral(ax, center=(0,0,0), turns=4, radius=1.2, color='#44ff88', lw=3):
    """🌀 Spiral"""
    t = np.linspace(0, turns*np.pi, 120)
    r = radius + 0.25*np.sin(6*t)
    x = r * np.cos(t) * np.sin(t*0.7) + center[0]
    y = r * np.sin(t) * np.cos(t*1.3) + center[1]
    z = 0.7 * np.sin(t*2.1) + center[2]
    ax.plot(x, y, z, color=color, lw=lw, alpha=0.9)

def draw_qec_vortex(ax, center=(0,0,0), major_r=1.1, minor_r=0.4, color='gold', lw=4):
    """⚛️ QEC VORTEX"""
    phi = np.linspace(0, 2*np.pi, 80)
    R = minor_r + 0.12*np.sin(9*phi)
    x = (major_r + R*np.cos(phi)) * np.cos(phi*0.4) + center[0]
    y = (major_r + R*np.cos(phi)) * np.sin(phi*0.4) + center[1]
    z = R * np.sin(phi) + center[2]
    ax.plot(x, y, z, color=color, lw=lw, alpha=0.85)

def draw_chaotic_shells(ax, scales=[0.9, 0.6, 0.35], alpha=0.3):
    """🔥 Kaotik Küreler"""
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:14j]
    for i, scale in enumerate(scales):
        distort = 0.18 * np.sin(7*u + i*12)
        x = scale * (1+distort) * np.cos(u) * np.sin(v)
        y = scale * (1+distort) * np.sin(u) * np.sin(v)
        z = scale * np.cos(v)
        ax.plot_surface(x, y, z, color=f'C{i}', alpha=alpha)

def draw_kececi_fractal_complete(ax, pulse_center=(0,0,0), pulse_r=0.3, frame=0):
    """🏆 Tam Fractal"""
    draw_kececi_spiral(ax)
    draw_qec_vortex(ax)
    draw_chaotic_shells(ax)
    u, v = np.mgrid[0:2*np.pi:22j, 0:np.pi:14j]
    pulse_rad = pulse_r + 0.15*np.sin(frame*0.3)
    x = pulse_rad*np.cos(u)*np.sin(v) + pulse_center[0]
    y = pulse_rad*np.sin(u)*np.sin(v) + pulse_center[1]
    z = pulse_rad*np.cos(v) + pulse_center[2]
    ax.plot_surface(x, y, z, color='cyan', alpha=0.75)

def kececi_3d_fractal(
    num_children: int = 8,
    max_level: int = 3,
    scale_factor: float = 0.4,
    base_radius: float = 1.0,
    min_radius: float = 0.05,
    color_scheme: str = "plasma",
    alpha_decay: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30.0,
    azim: float = 45.0,
    background_color: Union[str, Tuple[float, float, float], None] = "#0a0a0a",
    show_grid: bool = True,
    grid_alpha: float = 0.1,
    title: Optional[str] = None,
    show_axis_labels: bool = False,
    axis_label_color: str = "white",
    interactive_info: bool = False,
    return_figure: bool = True,  # YENİ: Figür döndürülsün mü?
    output_mode: str = "show",  # 'show', 'save', 'return'
    filename: str = "kececi_fractal_3d",
    dpi: int = 300,
    verbose: bool = True,
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Generates and visualizes 3D Keçeci fractals.
    
    Parameters:
    -----------
    num_children : int
        Number of child spheres at each level (default: 8)
    max_level : int
        Maximum recursion depth (default: 3)
    scale_factor : float
        Size reduction factor for child spheres (default: 0.4)
    base_radius : float
        Radius of the central sphere (default: 1.0)
    min_radius : float
        Minimum sphere radius (stops recursion when reached) (default: 0.05)
    color_scheme : str
        Matplotlib colormap name (default: 'plasma')
    alpha_decay : float
        Alpha transparency decay factor per level (default: 0.7)
    figsize : Tuple[int, int]
        Figure size (width, height) (default: (12, 10))
    elev : float
        Elevation angle for 3D view (default: 30)
    azim : float
        Azimuth angle for 3D view (default: 45)
    background_color : str or tuple
        Background color (default: '#0a0a0a')
    show_grid : bool
        Show grid lines (default: True)
    grid_alpha : float
        Grid transparency (default: 0.1)
    title : str or None
        Custom title (auto-generated if None)
    show_axis_labels : bool
        Show X, Y, Z axis labels (default: False)
    axis_label_color : str
        Color for axis labels (default: 'white')
    interactive_info : bool
        Show interactive instructions (default: False)
    return_figure : bool
        Return (fig, ax) tuple instead of showing/saving (default: True)
    output_mode : str
        'show', 'save', or 'return' (default: 'show')
    filename : str
        Base filename for saving (default: 'kececi_fractal_3d')
    dpi : int
        DPI for saved images (default: 300)
    verbose : bool
        Print progress information (default: True)
    
    Returns:
    --------
    None or Tuple[plt.Figure, plt.Axes]
        Depending on return_figure and output_mode parameters
    """
    
    if not HAS_3D:
        if verbose:
            print("Error: 3D plotting not available. Install matplotlib with 3D support.", 
                  file=sys.stderr)
        return None if not return_figure else (None, None)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Set background color
    bg_color = _parse_color(background_color) or (0.04, 0.04, 0.04)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Create color function
    cmap = get_cmap_safe(color_scheme)
    
    def color_func(level: int) -> Tuple[float, float, float, float]:
        """Returns color for a given level based on colormap."""
        return cmap(level / max(max_level, 1))
    
    # Generate the fractal
    center = np.array([0.0, 0.0, 0.0])
    
    if verbose:
        print("Generating 3D fractal...")
        print(f"   • Level: {max_level}")
        print(f"   • Children: {num_children}")
        print(f"   • Color scheme: {color_scheme}")
    
    _generate_recursive_3d_fractal(
        ax,
        center,
        base_radius,
        0,
        max_level,
        num_children,
        scale_factor,
        min_radius,
        color_func,
        alpha_decay,
    )
    
    # Set plot limits
    max_extent = base_radius * (1 + 2 * scale_factor * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    
    # Configure view
    ax.view_init(elev=elev, azim=azim)
    
    # Grid settings
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)
    
    # Axis labels
    if show_axis_labels:
        ax.set_xlabel("X", fontsize=10, labelpad=10, color=axis_label_color)
        ax.set_ylabel("Y", fontsize=10, labelpad=10, color=axis_label_color)
        ax.set_zlabel("Z", fontsize=10, labelpad=10, color=axis_label_color)
        
        ax.xaxis.label.set_color(axis_label_color)
        ax.yaxis.label.set_color(axis_label_color)
        ax.zaxis.label.set_color(axis_label_color)
        ax.tick_params(axis="x", colors=axis_label_color, labelsize=8)
        ax.tick_params(axis="y", colors=axis_label_color, labelsize=8)
        ax.tick_params(axis="z", colors=axis_label_color, labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    # Title
    if title is None:
        title = f"3D Keçeci Fractal (Levels: {max_level}, Children: {num_children})"
    
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=20)
    
    # Interactive info
    if interactive_info:
        info_text = (
            "Rotate: Left click + drag\n"
            "Zoom: Mouse wheel\n"
            "Pan: Right click + drag"
        )
        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )
    
    plt.tight_layout()
    
    # Output handling
    output_mode = output_mode.lower().strip()
    
    if output_mode == "show":
        plt.show()
        if return_figure:
            return fig, ax
        else:
            plt.close(fig)
            return None
    
    elif output_mode == "save":
        if filename:
            output_filename = f"{filename}.png"
            try:
                save_kwargs = {
                    "bbox_inches": "tight",
                    "pad_inches": 0.1,
                    "facecolor": fig.get_facecolor(),
                    "dpi": dpi,
                }
                plt.savefig(output_filename, **save_kwargs)
                if verbose:
                    print(f"3D Fractal saved to: '{os.path.abspath(output_filename)}'")
            except Exception as e:
                print(f"Error saving file: {e}", file=sys.stderr)
            finally:
                if not return_figure:
                    plt.close(fig)
        if return_figure:
            return fig, ax
        else:
            plt.close(fig)
            return None
    
    elif output_mode == "return" or return_figure:
        # Just return the figure without showing
        return fig, ax
    
    else:
        print(f"Invalid output_mode: '{output_mode}'. Choose 'show', 'save', or 'return'.",
              file=sys.stderr)
        plt.close(fig)
        return None

"""
def kececi_3d_fractal(
    num_children: int = 8,
    max_level: int = 3,
    scale_factor: float = 0.4,
    base_radius: float = 1.0,
    min_radius: float = 0.05,
    color_scheme: str = "plasma",
    alpha_decay: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30.0,
    azim: float = 45.0,
    background_color: Union[str, Tuple[float, float, float], None] = "#0a0a0a",
    show_grid: bool = True,
    grid_alpha: float = 0.1,
    title: Optional[str] = None,
    interactive: bool = False,  # Jupyter'da interactive=False yapıyoruz
    save_filename: Optional[str] = None,
    dpi: int = 150,
):

    #3D Keçeci fraktalı oluşturur ve görselleştirir.


    if not HAS_3D:
        print("Hata: 3D grafik desteği yok. Lütfen matplotlib 3D modülünü yükleyin.")
        return None, None

    # Figür oluştur
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Arkaplan rengini ayarla
    bg_color = _parse_color(background_color) or (0.04, 0.04, 0.04)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(bg_color)

    # Renk fonksiyonunu oluştur
    color_func = generate_color_function(color_scheme, max_level)

    # Fraktalı oluştur
    center = np.array([0.0, 0.0, 0.0])
    print(f"3D fraktal oluşturuluyor...")
    print(f"   • Seviye: {max_level}")
    print(f"   • Çocuk sayısı: {num_children}")
    print(f"   • Renk şeması: {color_scheme}")

    _generate_recursive_3d_fractal(
        ax,
        center,
        base_radius,
        0,
        max_level,
        num_children,
        scale_factor,
        min_radius,
        color_func,
        alpha_decay,
    )

    # Grafik sınırlarını ayarla
    max_extent = base_radius * (1 + 2 * scale_factor * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Görünüm açılarını ayarla
    ax.view_init(elev=elev, azim=azim)

    # Eksen etiketlerini ve ızgarayı ayarla
    ax.set_xlabel("X", fontsize=10, labelpad=10, color="white")
    ax.set_ylabel("Y", fontsize=10, labelpad=10, color="white")
    ax.set_zlabel("Z", fontsize=10, labelpad=10, color="white")

    # Eksen rengini ayarla
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.tick_params(axis="y", colors="white", labelsize=8)
    ax.tick_params(axis="z", colors="white", labelsize=8)

    # Izgara ayarları
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)

    # Başlık ekle
    if title is None:
        title = f"3D Keçeci Fraktalı | Seviye: {max_level} | Çocuk: {num_children}"

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=20)

    # Unicode karakterleri temizleyen basit bir info text
    if interactive:
        info_text = (
            "Fare ile döndür: Sol tık + sürükle\n"
            "Yakınlaştır/Uzaklaştır: Fare tekerleği\n"
            "Kaydır: Sağ tık + sürükle"
        )
        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    # Grafik düzenini ayarla
    plt.tight_layout()

    # Kaydetme
    if save_filename:
        try:
            # Font uyarılarını geçici olarak gizle
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                plt.savefig(
                    save_filename,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                    edgecolor="none",
                )
            print(f"Fraktal kaydedildi: {save_filename}")
        except Exception as e:
            print(f"Kaydetme hatası: {e}")

    print("3D fraktal hazır!")
    return fig, ax
"""


def _draw_recursive_circles(
    ax, x, y, radius, level, max_level, num_children, min_radius, scale_factor
):
    """
    Internal recursive helper function to draw child circles for general fractals.
    Not intended for direct use.
    """
    if level > max_level:
        return

    child_radius = radius * scale_factor
    if child_radius < min_radius:
        return

    distance_from_parent_center = radius - child_radius

    for i in range(num_children):
        angle_rad = np.deg2rad(360 / num_children * i)
        child_x = x + distance_from_parent_center * np.cos(angle_rad)
        child_y = y + distance_from_parent_center * np.sin(angle_rad)

        child_color = random_soft_color()
        # General-purpose fractal uses lw=0 for solid, borderless circles.
        _draw_circle_patch(
            ax, (child_x, child_y), child_radius, face_color=child_color, lw=0
        )

        try:
            _draw_recursive_circles(
                ax,
                child_x,
                child_y,
                child_radius,
                level + 1,
                max_level,
                num_children,
                min_radius,
                scale_factor,
            )
        except RecursionError:
            print(
                "Warning: Maximum recursion depth reached. Fractal may be incomplete.",
                file=sys.stderr,
            )
            return


def kececifractals_circle(
    initial_children: int = 6,
    recursive_children: int = 6,
    text: str = "Keçeci Fractals",
    font_size: int = 14,
    font_color: str = "black",
    font_style: str = "bold",
    font_family: str = "Arial",
    max_level: int = 4,
    min_size_factor: float = 0.001,
    scale_factor: float = 0.5,
    base_radius: float = 4.0,
    background_color: Union[str, Tuple[float, float, float], None] = None,
    initial_circle_color: Union[str, Tuple[float, float, float], None] = None,
    output_mode: str = "show",
    filename: str = "kececi_fractal_circle",
    dpi: int = 300,
) -> None:
    """
    Generates, displays, or saves a general-purpose, aesthetic Keçeci-style circle fractal.

    Args:
        initial_children: Number of first-level child circles
        recursive_children: Number of children for deeper levels
        text: Text to display around the fractal
        font_size: Font size for text
        font_color: Color of text (string or hex)
        font_style: Font style ('normal', 'bold', 'italic', etc.)
        font_family: Font family name
        max_level: Maximum recursion depth
        min_size_factor: Minimum radius as factor of base_radius
        scale_factor: Size reduction factor for child circles
        base_radius: Radius of the central circle
        background_color: Background color (hex string, named color, or RGB tuple)
        initial_circle_color: Color of central circle (hex string, named color, or RGB tuple)
        output_mode: 'show' or file format ('png', 'jpg', etc.)
        filename: Base filename for saving
        dpi: DPI for saved images
    """
    # Input validation
    if not isinstance(max_level, int) or max_level < 0:
        print("Error: max_level must be a non-negative integer.", file=sys.stderr)
        return
    if not (0 < scale_factor < 1):
        print(
            "Error: scale_factor must be a number between 0 and 1 (exclusive).",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Parse colors (accepts hex strings, named colors, or RGB tuples)
    bg_color = _parse_color(background_color) or random_soft_color()
    main_color = _parse_color(initial_circle_color) or random_soft_color()

    # Parse font color
    parsed_font_color = _parse_color(font_color) or (0, 0, 0)

    fig.patch.set_facecolor(bg_color)

    # Draw the main circle
    _draw_circle_patch(ax, (0, 0), base_radius, face_color=main_color, lw=0)

    min_absolute_radius = base_radius * min_size_factor
    limit = base_radius + 1.0

    # Text placement
    if text and isinstance(text, str) and len(text) > 0:
        text_radius = base_radius + 0.8
        for i, char in enumerate(text):
            angle_deg = (360 / len(text) * i) - 90
            angle_rad = np.deg2rad(angle_deg)
            x_text, y_text = text_radius * np.cos(angle_rad), text_radius * np.sin(
                angle_rad
            )
            ax.text(
                x_text,
                y_text,
                char,
                fontsize=font_size,
                ha="center",
                va="center",
                color=parsed_font_color,
                fontweight=font_style,
                fontfamily=font_family,
                rotation=angle_deg + 90,
            )
        limit = max(limit, text_radius + font_size * 0.1)

    # Start the recursion
    if max_level >= 1:
        initial_radius = base_radius * scale_factor
        if initial_radius >= min_absolute_radius:
            dist_initial = base_radius - initial_radius
            for i in range(initial_children):
                angle_rad = np.deg2rad(360 / initial_children * i)
                ix, iy = dist_initial * np.cos(angle_rad), dist_initial * np.sin(
                    angle_rad
                )
                i_color = random_soft_color()
                _draw_circle_patch(
                    ax, (ix, iy), initial_radius, face_color=i_color, lw=0
                )
                _draw_recursive_circles(
                    ax,
                    ix,
                    iy,
                    initial_radius,
                    2,
                    max_level,
                    recursive_children,
                    min_absolute_radius,
                    scale_factor,
                )

    # Plot adjustments
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plot_title = f"Keçeci Fractals ({text})" if text else "Keçeci Circle Fractal"
    plt.title(plot_title, fontsize=16)

    # Output handling
    output_mode = output_mode.lower().strip()
    if output_mode == "show":
        plt.show()
    elif output_mode in ["png", "jpg", "jpeg", "svg"]:
        output_filename = f"{filename}.{output_mode}"
        try:
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "facecolor": fig.get_facecolor(),
            }
            if output_mode in ["png", "jpg", "jpeg"]:
                save_kwargs["dpi"] = dpi
            plt.savefig(output_filename, format=output_mode, **save_kwargs)
            print(
                f"Fractal successfully saved to: '{os.path.abspath(output_filename)}'"
            )
        except Exception as e:
            print(
                f"Error: Could not save file '{output_filename}': {e}", file=sys.stderr
            )
        finally:
            plt.close(fig)
    else:
        print(
            f"Error: Invalid output_mode '{output_mode}'. Choose 'show', 'png', 'jpg', or 'svg'.",
            file=sys.stderr,
        )
        plt.close(fig)


# ==============================================================================
# PART 2: QUANTUM ERROR CORRECTION (QEC) VISUALIZATION
# ==============================================================================


def _draw_recursive_qec(
    ax,
    x,
    y,
    radius,
    level,
    max_level,
    num_children,
    scale_factor,
    physical_qubit_color,
    error_color,
    error_qubits,
    current_path,
):
    """
    Internal recursive function to draw physical qubits and check for errors for the QEC model.
    """
    if level > max_level:
        return

    child_radius = radius * scale_factor
    distance_from_parent_center = radius * (1 - scale_factor)

    for i in range(num_children):
        child_path = current_path + [i]
        angle_rad = np.deg2rad(360 / num_children * i)
        child_x = x + distance_from_parent_center * np.cos(angle_rad)
        child_y = y + distance_from_parent_center * np.sin(angle_rad)

        qubit_color = (
            error_color if child_path in error_qubits else physical_qubit_color
        )
        _draw_circle_patch(
            ax, (child_x, child_y), child_radius, face_color=qubit_color, lw=0.75
        )

        _draw_recursive_qec(
            ax,
            child_x,
            child_y,
            child_radius,
            level + 1,
            max_level,
            num_children,
            scale_factor,
            physical_qubit_color,
            error_color,
            error_qubits,
            child_path,
        )


def visualize_qec_fractal(
    physical_qubits_per_level: int = 5,
    recursion_level: int = 1,
    error_qubits: Optional[List[List[int]]] = None,
    logical_qubit_color: str = "#4A90E2",  # Blue
    physical_qubit_color: str = "#E0E0E0",  # Light Gray
    error_color: str = "#D0021B",  # Red
    background_color: str = "#1C1C1C",  # Dark Gray
    scale_factor: float = 0.5,
    filename: str = "qec_fractal_visualization",
    dpi: int = 300,
) -> None:
    """
    Visualizes a Quantum Error Correction (QEC) code concept using Keçeci Fractals.
    """
    error_qubits = [] if error_qubits is None else error_qubits

    fig, ax = plt.subplots(figsize=(12, 12))

    # Parse colors for QEC visualization
    logical_color_parsed = _parse_color(logical_qubit_color) or (
        0.29,
        0.56,
        0.89,
    )  # Default blue
    physical_color_parsed = _parse_color(physical_qubit_color) or (
        0.88,
        0.88,
        0.88,
    )  # Default light gray
    error_color_parsed = _parse_color(error_color) or (0.82, 0.01, 0.11)  # Default red
    bg_color_parsed = _parse_color(background_color) or (
        0.11,
        0.11,
        0.11,
    )  # Default dark gray

    fig.patch.set_facecolor(bg_color_parsed)

    base_radius = 5.0

    # Draw the Logical Qubit
    _draw_circle_patch(ax, (0, 0), base_radius, face_color=logical_color_parsed, lw=1.5)
    ax.text(
        0,
        0,
        "L",
        color="white",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        fontfamily="sans-serif",
    )

    # Draw the Physical Qubits
    if recursion_level >= 1:
        initial_radius = base_radius * scale_factor
        dist_initial = base_radius * (1 - scale_factor)
        for i in range(physical_qubits_per_level):
            child_path = [i]
            angle_rad = np.deg2rad(360 / physical_qubits_per_level * i)
            ix, iy = dist_initial * np.cos(angle_rad), dist_initial * np.sin(angle_rad)
            qubit_color = (
                error_color_parsed
                if child_path in error_qubits
                else physical_color_parsed
            )

            _draw_circle_patch(
                ax, (ix, iy), initial_radius, face_color=qubit_color, lw=0.75
            )
            # Add a number label to the first-level qubits for clarity
            label_color = "black" if qubit_color != error_color_parsed else "white"
            ax.text(
                ix,
                iy,
                str(i),
                color=label_color,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

            _draw_recursive_qec(
                ax,
                ix,
                iy,
                initial_radius,
                2,
                recursion_level,
                physical_qubits_per_level,
                scale_factor,
                physical_color_parsed,
                error_color_parsed,
                error_qubits,
                child_path,
            )

    # Finalize and Save the Plot
    ax.set_xlim(-base_radius - 1.5, base_radius + 1.5)
    ax.set_ylim(-base_radius - 1.5, base_radius + 1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    title = f"QEC Fractal Model: {physical_qubits_per_level}-Qubit Code | Level: {recursion_level} | Errors: {len(error_qubits)}"
    plt.title(title, color="white", fontsize=18, pad=20)

    output_filename = f"{filename}.png"
    plt.savefig(
        output_filename,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Visualization saved to: '{os.path.abspath(output_filename)}'")


# ==============================================================================
# PART 3: 3D KEÇECİ FRACTALS
# ==============================================================================

try:
    from mpl_toolkits.mplot3d import Axes3D, art3d

    HAS_3D = True
except ImportError:
    HAS_3D = False
    print(
        "Warning: 3D plotting not available. Install matplotlib for 3D support.",
        file=sys.stderr,
    )


def _draw_3d_sphere(ax, center, radius, color, alpha=1.0):
    """
    Draws a 3D sphere on the given axes.
    """
    if not HAS_3D:
        return

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        edgecolor="none",
        antialiased=True,
        shade=True,
    )


def _generate_recursive_3d_fractal(
    ax,
    center,
    radius,
    level,
    max_level,
    num_children,
    scale_factor,
    min_radius,
    color_func,
    alpha_decay,
):
    """
    Recursive function to generate 3D fractal spheres.
    """
    if level > max_level or radius < min_radius:
        return

    # Draw current sphere
    color = color_func(level)
    alpha = 1.0 * (alpha_decay**level)
    _draw_3d_sphere(ax, center, radius, color, alpha)

    # Calculate positions for child spheres
    child_radius = radius * scale_factor
    if child_radius < min_radius:
        return

    # For 3D, distribute children on a sphere surface
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(num_children):
        # Fibonacci sphere distribution for even spacing
        y = 1 - (i / float(num_children - 1)) * 2
        radius_xy = np.sqrt(1 - y * y)

        theta = phi * i

        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy

        # Scale to put children on surface of parent sphere
        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)

        child_center = center + direction * (radius + child_radius)

        # Recursive call
        _generate_recursive_3d_fractal(
            ax,
            child_center,
            child_radius,
            level + 1,
            max_level,
            num_children,
            scale_factor,
            min_radius,
            color_func,
            alpha_decay,
        )


def get_cmap_safe(cmap_name: str):
    """Güvenli colormap alımı, tüm matplotlib sürümleriyle uyumlu"""
    try:
        # Matplotlib 3.7+ için modern yöntem
        return plt.colormaps[cmap_name]
    except (AttributeError, KeyError):
        try:
            # Klasik yöntem
            return plt.get_cmap(cmap_name)
        except:
            # Son çare olarak plt.cm
            import matplotlib.cm as cm

            return cm.get_cmap(cmap_name)


def kececifractals_3d(
    num_children: int = 8,
    max_level: int = 3,
    scale_factor: float = 0.4,
    base_radius: float = 1.0,
    min_radius: float = 0.05,
    color_scheme: str = "plasma",
    alpha_decay: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30,
    azim: float = 45,
    output_mode: str = "show",
    filename: str = "kececi_fractal_3d",
    dpi: int = 300,
) -> None:
    """
    Generates a 3D version of Keçeci fractals.

    Args:
        num_children: Number of child spheres at each level
        max_level: Maximum recursion depth
        scale_factor: Size reduction factor for child spheres
        base_radius: Radius of the central sphere
        min_radius: Minimum sphere radius (stops recursion when reached)
        color_scheme: Matplotlib colormap name
        alpha_decay: Alpha transparency decay factor per level
        figsize: Figure size (width, height)
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        output_mode: 'show' or file format ('png', 'jpg', etc.)
        filename: Base filename for saving
        dpi: DPI for saved images
    """
    if not HAS_3D:
        print(
            "Error: 3D plotting not available. Install matplotlib with 3D support.",
            file=sys.stderr,
        )
        return

    # generate figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Set dark background for better contrast
    dark_bg = _parse_color("#0a0a0a") or (0.04, 0.04, 0.04)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    # Color function based on level - using def instead of lambda
    cmap = get_cmap_safe(color_scheme)

    def color_func(level: int) -> Tuple[float, float, float, float]:
        """Returns color for a given level based on colormap."""
        return cmap(level / max(max_level, 1))

    # generate the fractal
    center = np.array([0.0, 0.0, 0.0])
    _generate_recursive_3d_fractal(
        ax,
        center,
        base_radius,
        0,
        max_level,
        num_children,
        scale_factor,
        min_radius,
        color_func,
        alpha_decay,
    )

    # Set plot limits
    max_extent = base_radius * (1 + 2 * scale_factor * max_level)
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Configure view
    ax.view_init(elev=elev, azim=azim)

    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Add title
    plt.title(
        f"3D Keçeci Fractal (Levels: {max_level}, Children: {num_children})",
        color="white",
        fontsize=14,
        pad=20,
    )

    # Add lighting effect (simulated with grid)
    ax.grid(True, alpha=0.1, linestyle="--", linewidth=0.5)

    # Output handling
    output_mode = output_mode.lower().strip()
    if output_mode == "show":
        plt.show()
    elif output_mode in ["png", "jpg", "jpeg", "svg"]:
        output_filename = f"{filename}.{output_mode}"
        try:
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "facecolor": fig.get_facecolor(),
            }
            if output_mode in ["png", "jpg", "jpeg"]:
                save_kwargs["dpi"] = dpi
            plt.savefig(output_filename, format=output_mode, **save_kwargs)
            print(
                f"3D Fractal successfully saved to: '{os.path.abspath(output_filename)}'"
            )
        except Exception as e:
            print(
                f"Error: Could not save file '{output_filename}': {e}", file=sys.stderr
            )
        finally:
            plt.close(fig)
    else:
        print(
            f"Error: Invalid output_mode '{output_mode}'. Choose 'show', 'png', 'jpg', or 'svg'.",
            file=sys.stderr,
        )
        plt.close(fig)


# ==============================================================================
# PART 4: STRATUM MODEL VISUALIZATION
# ==============================================================================


def _draw_recursive_stratum_circles(
    ax,
    cx,
    cy,
    radius,
    level,
    max_level,
    state_collection,
    branching_rule_func,
    node_properties_func,
):
    """
    Internal recursive helper to draw the Stratum Circular Fractal.
    It uses provided functions for branching and node properties. Not for direct use.
    """
    if level >= max_level:
        return

    # Draw the main circle representing the quantum state
    level_color = plt.cm.plasma(level / max_level)
    ax.add_patch(
        plt.Circle((cx, cy), radius, facecolor=level_color, alpha=0.2, zorder=level)
    )

    # Get node properties using the PASSED-IN function
    node_props = node_properties_func(level, 0)
    ax.plot(
        cx,
        cy,
        "o",
        markersize=node_props.get("size", 10),
        color="white",
        alpha=0.8,
        zorder=level + max_level,
    )

    # Add this state's data to our collection
    state_collection.append(
        {
            "id": len(state_collection),
            "level": level,
            "energy": node_props.get("energy", 0.0),
            "size": node_props.get("size", 10),
            "color": level_color,
        }
    )

    # Determine the number of child states using the PASSED-IN function
    num_children = branching_rule_func(level)

    # Position and draw the child circles
    scale_factor = 0.5
    child_radius = radius * scale_factor
    distance_from_center = radius * (1 - scale_factor)

    for i in range(num_children):
        angle = 2 * math.pi * i / num_children + random.uniform(-0.1, 0.1)
        child_cx = cx + distance_from_center * math.cos(angle)
        child_cy = cy + distance_from_center * math.sin(angle)

        _draw_recursive_stratum_circles(
            ax,
            child_cx,
            child_cy,
            child_radius,
            level + 1,
            max_level,
            state_collection,
            branching_rule_func,
            node_properties_func,
        )


def visualize_stratum_model(
    ax,
    max_level,
    branching_rule_func,
    node_properties_func,
    initial_radius=100,
    start_cx=0,
    start_cy=0,
):
    """
    Public-facing function to visualize the Stratum Model as a circular fractal.
    This is the main entry point from your script.

    Args:
        ax: The matplotlib axes object to draw on.
        max_level (int): The maximum recursion depth.
        branching_rule_func (function): A function that takes a level (int) and returns the number of branches.
        node_properties_func (function): A function that takes a level and branch_index and returns a dict of properties (e.g., {'size': ..., 'energy': ...}).
        initial_radius (float): The radius of the first circle.
        start_cx, start_cy (float): The center coordinates of the first circle.

    Returns:
        list: A list of dictionaries, where each dictionary represents a generated state.
    """
    state_collection = []
    _draw_recursive_stratum_circles(
        ax,
        start_cx,
        start_cy,
        initial_radius,
        0,
        max_level,
        state_collection,
        branching_rule_func,
        node_properties_func,
    )
    return state_collection


def visualize_sequential_spectrum(ax, state_collection):
    """
    Draws all collected quantum states in a sequential spectrum using the Keçeci Layout,
    including dotted lines to show the connection between consecutive states.
    """
    if not state_collection:
        ax.text(0.5, 0.5, "No Data Available", color="white", ha="center", va="center")
        return

    G = nx.Graph()
    for state_data in state_collection:
        G.add_node(state_data["id"], **state_data)

    if len(G.nodes()) > 1:
        for i in range(len(G.nodes()) - 1):
            G.add_edge(i, i + 1)

    pos = kl.kececi_layout(
        G, primary_direction="top_down", primary_spacing=1.5, secondary_spacing=1.0
    )

    node_ids = list(G.nodes())
    node_sizes = [G.nodes[n].get("size", 10) * 5 for n in node_ids]
    node_colors = [G.nodes[n].get("color", "blue") for n in node_ids]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.5,
        ax=ax,
    )

    nx.draw_networkx_edges(G, pos, ax=ax, style="dotted", edge_color="gray", alpha=0.7)

    ax.set_title(
        "Sequential State Spectrum (Keçeci Layout)", color="white", fontsize=12
    )
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")


def generate_color_function(
    cmap_name: str, max_level: int
) -> Callable[[int], Tuple[float, float, float, float]]:
    """
    generates a color function that returns colors based on level.

    Args:
        cmap_name: Name of the matplotlib colormap
        max_level: Maximum level for normalization

    Returns:
        Function that takes a level and returns RGBA color
    """
    cmap = get_cmap_safe(cmap_name)

    def color_func(level: int) -> Tuple[float, float, float, float]:
        """Returns color for a given level based on colormap."""
        return cmap(level / max(max_level, 1))

    return color_func


def optimized_3d_fractal(
    num_children: int = 6,
    max_level: int = 3,
    resolution: int = 15,  # Düşük çözünürlük için
    show_plot: bool = True,
):
    """
    Optimize edilmiş 3D fraktal (hızlı render için).
    """
    if not HAS_3D:
        return None, None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Basit renk fonksiyonu
    def simple_color_func(level):
        colors = [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2)]
        return colors[level % len(colors)]

    # Optimize edilmiş küre çizimi
    def draw_sphere_fast(ax, center, radius, color, alpha=0.7):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none", shade=True)

    # Optimize edilmiş özyineleme
    def generate_fractal_fast(
        ax, center, radius, level, max_level, num_children, scale_factor
    ):
        if level > max_level or radius < 0.05:
            return

        # Küreyi çiz
        color = simple_color_func(level)
        draw_sphere_fast(ax, center, radius, color, alpha=0.7 - level * 0.15)

        # Çocuk küreler
        child_radius = radius * scale_factor

        for i in range(num_children):
            angle = 2 * np.pi * i / num_children
            elevation = np.pi * (i % 2) / 2  # Alternatif yükseklik

            x = np.cos(angle) * np.cos(elevation)
            y = np.sin(angle) * np.cos(elevation)
            z = np.sin(elevation)

            direction = np.array([x, y, z])
            direction = direction / np.linalg.norm(direction)

            child_center = center + direction * (radius + child_radius)

            generate_fractal_fast(
                ax,
                child_center,
                child_radius,
                level + 1,
                max_level,
                num_children,
                scale_factor,
            )

    # Fraktalı oluştur
    center = np.array([0.0, 0.0, 0.0])
    generate_fractal_fast(ax, center, 1.0, 0, max_level, num_children, 0.4)

    # Görünüm ayarları
    max_extent = 1.0 * (1 + 2 * 0.4 * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=25, azim=45)

    ax.set_facecolor("#0a0a0a")
    ax.grid(True, alpha=0.1)
    ax.set_title(
        f"Hızlı 3D Fraktal (Çözünürlük: {resolution})", color="white", fontsize=12
    )

    if show_plot:
        plt.tight_layout()
        plt.show()

    return fig, ax


# Her fraktal için ayrı figür oluştur, sonra birleştir
def generate_single_fractal(num_children, max_level, color_scheme, title):
    """Tek bir fraktal oluşturur ve surface objelerini döndürür."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    color_func = generate_color_function(color_scheme, max_level)
    center = np.array([0.0, 0.0, 0.0])

    # Fraktalı oluştur
    _generate_recursive_3d_fractal(
        ax, center, 1.0, 0, max_level, num_children, 0.4, 0.05, color_func, 0.7
    )

    # Görünüm ayarları
    max_extent = 1.0 * (1 + 2 * 0.4 * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=25, azim=45)
    ax.set_title(title, fontsize=10, color="white", pad=10)
    ax.set_facecolor("#0a0a0a")
    ax.grid(True, alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Surface objelerini topla (list() kullanarak)
    surfaces = list(ax.collections)

    plt.close(fig)
    return surfaces


def generate_fractal_directly(ax, config: dict):
    """
    Fraktalı doğrudan verilen Matplotlib ekseninde oluşturur.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D eksen objesi
    config : dict
        Fraktal konfigürasyonu:
        - num_children: Her seviyedeki çocuk sayısı
        - max_level: Maksimum özyineleme seviyesi
        - color_scheme: Renk şeması ismi
        - title: (opsiyonel) Başlık
        - scale_factor: (opsiyonel) Ölçek faktörü, varsayılan 0.4
        - base_radius: (opsiyonel) Ana yarıçap, varsayılan 1.0
        - min_radius: (opsiyonel) Minimum yarıçap, varsayılan 0.05
        - alpha_decay: (opsiyonel) Alpha azalma faktörü, varsayılan 0.7

    Returns:
    --------
    None

    Raises:
    -------
    ThreeDNotSupportedError
        3D grafik desteği yoksa
    InvalidAxisError
        Eksen 3D değilse veya geçersizse
    FractalParameterError
        Konfigürasyon parametreleri geçersizse
    """
    # 3D desteği kontrolü
    if not HAS_3D:
        raise ThreeDNotSupportedError(
            "3D grafik desteği yok. Lütfen matplotlib'in 3D modülünü yükleyin."
        )

    # Eksen kontrolü
    try:
        # Eksenin 3D olup olmadığını kontrol et
        if not hasattr(ax, "get_proj"):
            raise InvalidAxisError("Verilen eksen 3D değil.")
    except AttributeError:
        raise InvalidAxisError("Geçersiz eksen objesi.")

    # Konfigürasyon validasyonu
    required_keys = ["num_children", "max_level", "color_scheme"]
    for key in required_keys:
        if key not in config:
            raise FractalParameterError(f"Gerekli parametre eksik: '{key}'")

    # Parametre validasyonu
    if not isinstance(config["num_children"], int) or config["num_children"] < 1:
        raise FractalParameterError("num_children pozitif bir tamsayı olmalıdır.")

    if not isinstance(config["max_level"], int) or config["max_level"] < 0:
        raise FractalParameterError("max_level negatif olmayan bir tamsayı olmalıdır.")

    # Varsayılan değerleri ayarla
    config.setdefault("scale_factor", 0.4)
    config.setdefault("base_radius", 1.0)
    config.setdefault("min_radius", 0.05)
    config.setdefault("alpha_decay", 0.7)

    # Parametre aralık kontrolü
    if not (0 < config["scale_factor"] < 1):
        raise FractalParameterError("scale_factor 0 ile 1 arasında olmalıdır.")

    if config["base_radius"] <= 0:
        raise FractalParameterError("base_radius pozitif olmalıdır.")

    if config["min_radius"] <= 0:
        raise FractalParameterError("min_radius pozitif olmalıdır.")

    if not (0 <= config["alpha_decay"] <= 1):
        raise FractalParameterError("alpha_decay 0 ile 1 arasında olmalıdır.")

    # Renk fonksiyonunu oluştur
    try:
        color_func = generate_color_function(
            config["color_scheme"], config["max_level"]
        )
    except Exception as e:
        raise FractalParameterError(f"Renk şeması oluşturulamadı: {e}")

    center = np.array([0.0, 0.0, 0.0])

    # Fraktalı oluştur
    try:
        _generate_recursive_3d_fractal(
            ax,
            center,
            config["base_radius"],
            0,
            config["max_level"],
            config["num_children"],
            config["scale_factor"],
            config["min_radius"],
            color_func,
            config["alpha_decay"],
        )
    except RecursionError:
        raise FractalParameterError(
            f"Özyineleme sınırı aşıldı. max_level değerini azaltmayı deneyin."
        )
    except Exception as e:
        raise KececiFractalError(f"Fraktal oluşturulurken hata: {e}")

    # Eksen ayarları
    max_extent = (
        config["base_radius"]
        * (1 + 2 * config["scale_factor"] * config["max_level"])
        * 1.2
    )
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=25, azim=45)

def generate_simple_3d_fractal(
    ax,
    num_children: int = 6,
    max_level: int = 3,
    color_scheme: str = "viridis",
    **kwargs,
):
    """
    Basit bir 3D fraktal oluşturur.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D eksen objesi
    num_children : int, optional
        Her seviyedeki çocuk sayısı (varsayılan: 6)
    max_level : int, optional
        Maksimum özyineleme seviyesi (varsayılan: 3)
    color_scheme : str, optional
        Renk şeması ismi (varsayılan: 'viridis')
    **kwargs : dict
        Ek parametreler:
        - scale_factor: Ölçek faktörü (varsayılan: 0.4)
        - base_radius: Ana yarıçap (varsayılan: 1.0)
        - min_radius: Minimum yarıçap (varsayılan: 0.05)
        - alpha_decay: Alpha azalma faktörü (varsayılan: 0.7)
        - elev: Görünüm eğim açısı (varsayılan: 25)
        - azim: Görünüm azimut açısı (varsayılan: 45)
        - show_grid: Izgara gösterilsin mi? (varsayılan: True)
        - grid_alpha: Izgara saydamlığı (varsayılan: 0.1)
        - background_color: Arkaplan rengi (varsayılan: '#0a0a0a')
        - title: Başlık (varsayılan: None)
        - title_size: Başlık font boyutu (varsayılan: 14)
        - title_weight: Başlık font kalınlığı (varsayılan: 'bold')
        - title_color: Başlık rengi (varsayılan: 'white')
        - title_pad: Başlık padding'i (varsayılan: 20)
        - show_axis_labels: Eksen etiketleri gösterilsin mi? (varsayılan: False)
        - xlabel: X eksen etiketi (varsayılan: 'X')
        - ylabel: Y eksen etiketi (varsayılan: 'Y')
        - zlabel: Z eksen etiketi (varsayılan: 'Z')
        - axis_label_color: Eksen etiketi rengi (varsayılan: 'white')
        - axis_label_size: Eksen etiketi boyutu (varsayılan: 10)
        - tick_color: Tick rengi (varsayılan: 'white')

    Returns:
    --------
    None

    Raises:
    -------
    ThreeDNotSupportedError
        3D grafik desteği yoksa
    InvalidAxisError
        Eksen 3D değilse veya geçersizse
    FractalParameterError
        Parametreler geçersizse
    """
    # 3D desteği kontrolü
    if not HAS_3D:
        raise ThreeDNotSupportedError(
            "3D grafik desteği yok. Lütfen matplotlib'in 3D modülünü yükleyin."
        )

    # Eksen kontrolü
    try:
        if not hasattr(ax, "get_proj"):
            raise InvalidAxisError("Verilen eksen 3D değil.")
    except AttributeError:
        raise InvalidAxisError("Geçersiz eksen objesi.")

    # Parametre validasyonu
    if not isinstance(num_children, int) or num_children < 1:
        raise FractalParameterError("num_children pozitif bir tamsayı olmalıdır.")

    if not isinstance(max_level, int) or max_level < 0:
        raise FractalParameterError("max_level negatif olmayan bir tamsayı olmalıdır.")

    # Varsayılan değerleri ayarla
    scale_factor = kwargs.get("scale_factor", 0.4)
    base_radius = kwargs.get("base_radius", 1.0)
    min_radius = kwargs.get("min_radius", 0.05)
    alpha_decay = kwargs.get("alpha_decay", 0.7)
    elev = kwargs.get("elev", 25)
    azim = kwargs.get("azim", 45)
    show_grid = kwargs.get("show_grid", True)
    grid_alpha = kwargs.get("grid_alpha", 0.1)
    background_color = kwargs.get("background_color", "#0a0a0a")
    title = kwargs.get("title", None)

    # Parametre aralık kontrolü
    if not (0 < scale_factor < 1):
        raise FractalParameterError("scale_factor 0 ile 1 arasında olmalıdır.")

    if base_radius <= 0:
        raise FractalParameterError("base_radius pozitif olmalıdır.")

    if min_radius <= 0:
        raise FractalParameterError("min_radius pozitif olmalıdır.")

    if not (0 <= alpha_decay <= 1):
        raise FractalParameterError("alpha_decay 0 ile 1 arasında olmalıdır.")

    if not (-90 <= elev <= 90):
        raise FractalParameterError("elev -90 ile 90 arasında olmalıdır.")

    if not (0 <= azim <= 360):
        raise FractalParameterError("azim 0 ile 360 arasında olmalıdır.")

    # Arkaplan rengini ayarla
    try:
        bg_color = _parse_color(background_color) or (0.04, 0.04, 0.04)
        ax.set_facecolor(bg_color)
    except Exception as e:
        raise ColorParseError(f"Arkaplan rengi parse edilemedi: {e}")

    # Renk fonksiyonunu oluştur
    try:
        color_func = generate_color_function(color_scheme, max_level)
    except Exception as e:
        raise FractalParameterError(f"Renk şeması oluşturulamadı: {e}")

    center = np.array([0.0, 0.0, 0.0])

    # Fraktalı oluştur
    try:
        _generate_recursive_3d_fractal(
            ax,
            center,
            base_radius,
            0,
            max_level,
            num_children,
            scale_factor,
            min_radius,
            color_func,
            alpha_decay,
        )
    except RecursionError:
        raise FractalParameterError(
            f"Özyineleme sınırı aşıldı. max_level değerini azaltmayı deneyin."
        )
    except Exception as e:
        raise KececiFractalError(f"Fraktal oluşturulurken hata: {e}")

    # Eksen ayarları
    max_extent = base_radius * (1 + 2 * scale_factor * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=elev, azim=azim)

    # Izgara ayarları
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)

    # Başlık ekle
    if title:
        ax.set_title(
            title,
            fontsize=kwargs.get("title_size", 14),
            fontweight=kwargs.get("title_weight", "bold"),
            color=kwargs.get("title_color", "white"),
            pad=kwargs.get("title_pad", 20),
        )

    # Eksen etiketleri
    if kwargs.get("show_axis_labels", False):
        ax.set_xlabel(
            kwargs.get("xlabel", "X"),
            color=kwargs.get("axis_label_color", "white"),
            fontsize=kwargs.get("axis_label_size", 10),
        )
        ax.set_ylabel(
            kwargs.get("ylabel", "Y"),
            color=kwargs.get("axis_label_color", "white"),
            fontsize=kwargs.get("axis_label_size", 10),
        )
        ax.set_zlabel(
            kwargs.get("zlabel", "Z"),
            color=kwargs.get("axis_label_color", "white"),
            fontsize=kwargs.get("axis_label_size", 10),
        )

        # Eksen etiketi renkleri
        ax.xaxis.label.set_color(kwargs.get("axis_label_color", "white"))
        ax.yaxis.label.set_color(kwargs.get("axis_label_color", "white"))
        ax.zaxis.label.set_color(kwargs.get("axis_label_color", "white"))

        # Tick renkleri
        ax.tick_params(axis="x", colors=kwargs.get("tick_color", "white"))
        ax.tick_params(axis="y", colors=kwargs.get("tick_color", "white"))
        ax.tick_params(axis="z", colors=kwargs.get("tick_color", "white"))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

def draw_kececi_internal_fractal3d(
    ax,
    center=(0, 0, 0),
    radius=1.0,
    depth=3,
    num_children=8,
    scale_factor=0.4,
    min_radius=0.02,
    current_depth=0
):
    """
    🌀 3D Keçeci Internal Fractal – Belirgin Renk Ayrımı ile
    
    Çocuk küreler, ebeveyn kürenin İÇ yüzeyine teğet olacak şekilde yerleştirilir.
    Her seviye sabit, kontrastlı bir renkle gösterilir.
    """
    if depth < 0 or radius < min_radius:
        return

    # Sabit, belirgin renk paleti (derinliğe göre)
    color_palette = [
        (0.9, 0.2, 0.2),   # depth 0: kırmızı (merkez)
        (0.9, 0.6, 0.2),   # depth 1: turuncu
        (0.9, 0.9, 0.2),   # depth 2: sarı
        (0.2, 0.9, 0.4),   # depth 3: yeşil
        (0.0, 0.8, 0.9),   # depth 4: turkuaz
        (0.3, 0.3, 0.9),   # depth 5: mavi
        (0.6, 0.2, 0.8),   # depth 6: mor
    ]
    color = color_palette[current_depth % len(color_palette)]

    # Ana küreyi çiz
    draw_sphere(
        ax,
        center=center,
        radius=radius,
        color=color,
        alpha=0.7 + 0.05 * current_depth,  # iç katmanlar daha az saydam
        edgecolor='none'
    )

    if depth == 0:
        return

    # Çocuk küreler: ebeveynin içine, iç yüzeye teğet
    child_radius = radius * scale_factor
    directions = get_icosahedron_vertices()[:num_children]
    # Merkezden mesafe = R_ebeveyn - R_çocuk → iç teğet
    distance_from_center = radius - child_radius

    for d in directions:
        child_center = np.array(center) + distance_from_center * d
        draw_kececi_internal_fractal_3d(
            ax,
            center=tuple(child_center),
            radius=child_radius,
            depth=depth - 1,
            num_children=num_children,
            scale_factor=scale_factor,
            min_radius=min_radius,
            current_depth=current_depth + 1
        )

def draw_kececi_internal_fractal_3d(
    ax,
    center=(0, 0, 0),
    radius=1.0,
    depth=3,
    num_children=8,
    scale_factor=0.4,
    min_radius=0.02,
    current_depth=0
):
    if depth < 0 or radius < min_radius:
        return

    color = HIGH_CONTRAST_COLORS[current_depth % len(HIGH_CONTRAST_COLORS)]

    draw_sphere(
        ax,
        center=center,
        radius=radius,
        color=color,
        alpha=0.30,
        edgecolor='none'
    )

    if depth == 0:
        return

    child_radius = radius * scale_factor
    directions = get_icosahedron_vertices()[:num_children]
    distance_from_center = radius - child_radius

    for d in directions:
        child_center = np.array(center) + distance_from_center * d
        draw_kececi_internal_fractal_3d(
            ax,
            center=tuple(child_center),
            radius=child_radius,
            depth=depth - 1,
            num_children=num_children,
            scale_factor=scale_factor,
            min_radius=min_radius,
            current_depth=current_depth + 1
        )

def draw_kececi_external_fractal3d(
    ax,
    center=(0, 0, 0),
    radius=1.0,
    depth=3,
    num_children=8,
    scale_factor=0.45,
    min_radius=0.02,
    current_depth=0
):
    """
    🌀 Keçeci External Fractal (3D)
    - Küreler dışa doğru büyür.
    - Her seviye belirgin renkle gösterilir.
    - Fiziksel olarak tutarlı (katı cisimler çarpışmaz).
    """
    if depth < 0 or radius < min_radius:
        return

    # Belirgin renk paleti (derinliğe göre)
    color_palette = [
        (0.1, 0.2, 0.8),   # depth 0: koyu mavi
        (0.0, 0.8, 0.9),   # depth 1: turkuaz
        (0.2, 0.9, 0.4),   # depth 2: parlak yeşil
        (0.9, 0.9, 0.2),   # depth 3: sarı
        (0.9, 0.3, 0.2),   # depth 4: turuncu-kırmızı
        (0.6, 0.1, 0.6),   # depth 5: mor
    ]
    color = color_palette[current_depth % len(color_palette)]

    # Ana küreyi çiz
    draw_sphere(
        ax, center, radius,
        color=color,
        alpha=0.75,
        edgecolor='none'
    )

    if depth == 0:
        return  # yaprak düğüm

    # Çocuk küreler: dışa doğru
    child_radius = radius * scale_factor
    directions = get_icosahedron_vertices()[:num_children]
    distance = radius + child_radius  # dışa temas

    for d in directions:
        child_center = np.array(center) + distance * d
        draw_kececi_external_fractal_3d(
            ax,
            center=tuple(child_center),
            radius=child_radius,
            depth=depth - 1,
            num_children=num_children,
            scale_factor=scale_factor,
            min_radius=min_radius,
            current_depth=current_depth + 1
        )

def draw_kececi_external_fractal_3d(
    ax,
    center=(0, 0, 0),
    radius=1.0,
    depth=3,
    num_children=8,
    scale_factor=0.45,
    min_radius=0.02,
    current_depth=0
):
    """
    🌀 Keçeci External Fractal (3D)
    - Küreler dışa doğru büyür.
    - Her seviye belirgin renkle gösterilir.
    - Fiziksel olarak tutarlı (katı cisimler çarpışmaz).
    """
    if depth < 0 or radius < min_radius:
        return

    color = HIGH_CONTRAST_COLORS[current_depth % len(HIGH_CONTRAST_COLORS)]

    # Ana küreyi çiz
    draw_sphere(
        ax, center, radius,
        color=color,
        alpha=0.75,
        edgecolor='none'
    )

    if depth == 0:
        return  # yaprak düğüm

    # Çocuk küreler: dışa doğru
    child_radius = radius * scale_factor
    directions = get_icosahedron_vertices()[:num_children]
    distance = radius + child_radius  # dışa temas

    for d in directions:
        child_center = np.array(center) + distance * d
        draw_kececi_external_fractal_3d(
            ax,
            center=tuple(child_center),
            radius=child_radius,
            depth=depth - 1,
            num_children=num_children,
            scale_factor=scale_factor,
            min_radius=min_radius,
            current_depth=current_depth + 1
        )

# ==============================================================================
# ÖRNEK KULLANIM FONKSİYONLARI (isteğe bağlı)
# ==============================================================================
def example_multiple_fractals():
    """
    Çoklu fraktal karşılaştırması örneği.

    Returns:
    --------
    matplotlib.figure.Figure or None
        Oluşturulan figür veya hata durumunda None
    """
    if not HAS_3D:
        print("Hata: 3D grafik desteği yok.")
        return None

    try:
        import matplotlib.pyplot as plt

        # Ana figür oluştur
        fig, axes = plt.subplots(
            2, 2, figsize=(15, 12), subplot_kw={"projection": "3d"}
        )
        fig.patch.set_facecolor("#111111")

        # Farklı parametre kombinasyonları
        configs = [
            {
                "num_children": 4,
                "max_level": 2,
                "color_scheme": "viridis",
                "title": "Küçük Fraktal",
            },
            {
                "num_children": 8,
                "max_level": 3,
                "color_scheme": "plasma",
                "title": "Orta Fraktal",
            },
            {
                "num_children": 12,
                "max_level": 3,
                "color_scheme": "summer",
                "title": "Yoğun Fraktal",
            },
            {
                "num_children": 6,
                "max_level": 4,
                "color_scheme": "cool",
                "title": "Derin Fraktal",
            },
        ]

        # Her fraktalı doğrudan kendi ekseninde oluştur
        for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
            try:
                generate_fractal_directly(ax, config)

                # Eksen görünüm ayarları
                ax.set_title(
                    config["title"],
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    pad=15,
                )
                ax.set_facecolor("#0a0a0a")
                ax.grid(True, alpha=0.15)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            except Exception as e:
                print(f"Fraktal {config['title']} oluşturulurken hata: {e}")
                # Hata durumunda boş bir metin göster
                ax.text(0.5, 0.5, 0.5, "Hata", color="red", ha="center", va="center")

        plt.suptitle(
            "Farklı 3D Keçeci Fraktal Çeşitleri",
            fontsize=16,
            fontweight="bold",
            color="white",
            y=0.95,
        )
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Çoklu fraktal örneği oluşturulurken hata: {e}")
        return None


def example_view_angles():
    """
    Farklı görünüm açıları örneği.

    Returns:
    --------
    matplotlib.figure.Figure or None
        Oluşturulan figür veya hata durumunda None
    """
    if not HAS_3D:
        print("Hata: 3D grafik desteği yok.")
        return None

    try:
        import matplotlib.pyplot as plt

        # Tek bir fraktal oluştur ve farklı açılardan göster
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor("#111111")

        # Fraktal parametreleri
        fractal_params = {
            "num_children": 8,
            "max_level": 3,
            "scale_factor": 0.4,
            "color_scheme": "hot",
        }

        # Tüm alt eksenlerde aynı fraktalı oluştur
        view_angles = [
            (30, 0, "Ön Görünüm"),
            (30, 90, "Sağ Görünüm"),
            (30, 180, "Arka Görünüm"),
            (30, 270, "Sol Görünüm"),
        ]

        for idx, (elev, azim, title) in enumerate(view_angles, 1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")

            # Fraktalı bu eksende oluştur
            try:
                generate_simple_3d_fractal(
                    ax,
                    num_children=fractal_params["num_children"],
                    max_level=fractal_params["max_level"],
                    color_scheme=fractal_params["color_scheme"],
                    scale_factor=fractal_params["scale_factor"],
                    elev=elev,
                    azim=azim,
                    title=f"{title}\n(elev={elev}°, azim={azim}°)",
                    title_size=10,
                    show_axis_labels=False,
                )

                # Ek ayarlar
                ax.set_facecolor("#0a0a0a")
                ax.grid(True, alpha=0.1)

            except Exception as e:
                print(f"Görünüm açısı {title} oluşturulurken hata: {e}")
                ax.text(0.5, 0.5, 0.5, "Hata", color="red", ha="center", va="center")

        plt.suptitle(
            "3D Keçeci Fraktalı - Farklı Görünüm Açıları",
            fontsize=14,
            fontweight="bold",
            color="white",
            y=0.95,
        )
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Görünüm açıları örneği oluşturulurken hata: {e}")
        return None


def example_simple_fractal():
    """
    Basit fraktal örneği.

    Returns:
    --------
    matplotlib.figure.Figure or None
        Oluşturulan figür veya hata durumunda None
    """
    if not HAS_3D:
        print("Hata: 3D grafik desteği yok.")
        return None

    try:
        import matplotlib.pyplot as plt

        # Yeni bir figür oluştur
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Fraktalı oluştur
        generate_simple_3d_fractal(
            ax,
            num_children=7,
            max_level=4,
            color_scheme="coolwarm",
            title="Basit 3D Keçeci Fraktalı",
            show_axis_labels=True,
            xlabel="X Ekseni",
            ylabel="Y Ekseni",
            zlabel="Z Ekseni",
        )

        # Figür arkaplan rengi
        fig.patch.set_facecolor("#111111")
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Basit fraktal örneği oluşturulurken hata: {e}")
        return None

class KececiFractalOpenCL:
    def __init__(self):
        platform = select_opencl_platform()
        self.device = platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self._compile_kernel()
        self.max_circles = 0
        self.circles_buf = None
        self.output_buf = None
        self.max_output_floats = 0

    def _compile_kernel(self):
        kernel_src = """
        typedef struct { float cx, cy, r; float r_col, g_col, b_col; } Circle;
        __kernel void circle_fractal(
            __global float* output, const int width, const int height,
            __global const Circle* circles, const int num_circles, const float limit)
        {
            int px = get_global_id(0), py = get_global_id(1);
            if (px >= width || py >= height) return;
            float x = -limit + 2.0f * limit * px / (float)(width - 1);
            float y = -limit + 2.0f * limit * py / (float)(height - 1);
            int best_idx = -1; float best_r = 1e9f;
            for (int i = 0; i < num_circles; i++) {
                Circle c = circles[i];
                float dx = x - c.cx, dy = y - c.cy;
                if (dx*dx + dy*dy <= c.r * c.r) {
                    if (c.r < best_r) { best_r = c.r; best_idx = i; }
                }
            }
            int idx = (py * width + px) * 3;
            if (best_idx >= 0) {
                Circle c = circles[best_idx];
                output[idx] = c.r_col; output[idx+1] = c.g_col; output[idx+2] = c.b_col;
            } else { output[idx] = 0.0f; output[idx+1]=0.0f; output[idx+2]=0.0f; }
        }
        """
        self.prg = cl.Program(self.ctx, kernel_src).build()
        self.knl = self.prg.circle_fractal

    def generate_circles_with_colors(self, base_radius, scale_factor,
                                     initial_children, recursive_children,
                                     max_level, min_size_factor, main_color):
        circles = []
        min_r = base_radius * min_size_factor
        def add(cx, cy, r, col):
            circles.append((cx, cy, r, col[0], col[1], col[2]))
        add(0.0, 0.0, base_radius, main_color)
        def recurse(cx, cy, r, level):
            if level > max_level: return
            cr = r * scale_factor
            if cr < min_r: return
            dist = r - cr
            n = initial_children if level == 1 else recursive_children
            for i in range(n):
                ang = 2 * np.pi * i / n
                ix = cx + dist * np.cos(ang)
                iy = cy + dist * np.sin(ang)
                col = random_soft_color() if random_soft_color else (random.uniform(0.4,0.95) for _ in range(3))
                if not isinstance(col, tuple): col = tuple(col)
                add(ix, iy, cr, col)
                recurse(ix, iy, cr, level+1)
        if max_level >= 1: recurse(0.0, 0.0, base_radius, 1)
        return circles

    def render_gpu(self, circles, width, height, limit):
        num_circles = len(circles)
        output_floats = width * height * 3
        # buffer yönetimi
        if self.circles_buf is None or num_circles > self.max_circles:
            if self.circles_buf: del self.circles_buf
            self.max_circles = max(num_circles, 1024)
            self.circles_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.max_circles * 6 * 4)
        if self.output_buf is None or output_floats > self.max_output_floats:
            if self.output_buf: del self.output_buf
            self.max_output_floats = output_floats
            self.output_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, output_floats * 4)

        circles_arr = np.array(circles, dtype=np.float32).flatten()
        cl.enqueue_copy(self.queue, self.circles_buf, circles_arr).wait()

        self.knl(self.queue, (width, height), None,
                 self.output_buf,
                 np.int32(width), np.int32(height),
                 self.circles_buf,
                 np.int32(num_circles),
                 np.float32(limit))
        output = np.zeros((height, width, 3), dtype=np.float32)
        cl.enqueue_copy(self.queue, output, self.output_buf).wait()
        return output

    def kececifractals_circle_gpu(self,
                              initial_children: int = 3,
                              recursive_children: int = 3,
                              text: str = "Keçeci Fractals",
                              font_size: int = 14,
                              font_color: str = "black",
                              font_style: str = "bold",
                              font_family: str = "Arial",
                              max_level: int = 4,
                              min_size_factor: float = 0.001,
                              scale_factor: float = 0.5,
                              base_radius: float = 12.0,
                              background_color=None,
                              initial_circle_color=None,
                              output_mode: str = "show",
                              filename: str = "kececi_fractal_circle_gpu",
                              dpi: int = 300,
                              width: int = 1024,
                              height: int = 1024,
                              view_limit: float = None):   # <-- YENİ
        if not isinstance(max_level, int) or max_level < 0:
            print("Error: max_level must be a non-negative integer.", file=sys.stderr)
            return
        if not (0 < scale_factor < 1):
            print("Error: scale_factor must be between 0 and 1.", file=sys.stderr)
            return

        # Ana daire rengi
        from matplotlib.colors import to_rgb
        if initial_circle_color is not None:
            main_color = to_rgb(initial_circle_color) if isinstance(initial_circle_color, str) else initial_circle_color
        else:
            main_color = random_soft_color()
    
        # Daireleri oluştur
        circles = self.generate_circles_with_colors(
            base_radius, scale_factor, initial_children, recursive_children,
            max_level, min_size_factor, main_color
        )
    
        # --- LİMİT HESAPLAMA ---
        if view_limit is not None:
            limit = view_limit
        else:
            limit = base_radius + 1.0
            if text and isinstance(text, str) and len(text) > 0:
                text_radius = base_radius + 0.8
                limit = max(limit, text_radius + font_size * 0.1)
            if circles:
                circles_arr = np.array(circles, dtype=np.float32)
                max_ext = circles_arr[:, 0:2].max() + circles_arr[:, 2].max()
                limit = max(limit, max_ext * 1.05)
    
        img = self.render_gpu(circles, width, height, limit)

        # Arka plan rengini uygula
        if background_color:
            bg_rgb = to_rgb(background_color) if isinstance(background_color, str) else background_color
            mask = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
            img[mask] = bg_rgb

        # Çizim
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[-limit, limit, -limit, limit], origin='lower')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axis('off')

        # Metin
        if text and isinstance(text, str) and len(text) > 0:
            text_radius = base_radius + 0.8
            fc = to_rgb(font_color) if isinstance(font_color, str) else (0,0,0)
            for i, char in enumerate(text):
                angle_deg = (360 / len(text) * i) - 90
                angle_rad = np.deg2rad(angle_deg)
                x_text = text_radius * np.cos(angle_rad)
                y_text = text_radius * np.sin(angle_rad)
                ax.text(x_text, y_text, char,
                        fontsize=font_size, ha="center", va="center",
                        color=fc, fontweight=font_style, fontfamily=font_family,
                        rotation=angle_deg + 90)

        plot_title = f"Keçeci Fractals GPU ({text})" if text else "Keçeci Circle Fractal GPU"
        plt.title(plot_title, fontsize=16)

        output_mode = output_mode.lower().strip()
        if output_mode == "show":
            plt.show()
        elif output_mode in ["png", "jpg", "jpeg", "svg"]:
            output_filename = f"{filename}.{output_mode}"
            plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
            print(f"Fractal saved to: {output_filename}")
            plt.close()
        else:
            print(f"Invalid output_mode: {output_mode}")
            plt.close()

class KececiFractalGPU:
    def __init__(self, prefer_rusticl=True):
        """ Bütün işletim sistemleri için ortaktır"""
        platform = select_opencl_platform(prefer_rusticl)
        self.device = platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self._compile_kernel()
        self.max_circles = 0
        self.circles_buf = None
        self.output_buf = None
        self.max_output_floats = 0

    """
    def __init__(self, platform_name="rusticl"):
        #Sadece Linux için
        platform = next(p for p in cl.get_platforms() if p.name.lower().startswith(platform_name))
        self.device = platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self._compile_kernel()
    """

    def _compile_kernel(self):
        kernel_src = """
        typedef struct {
            float cx, cy, r;
            float r_col, g_col, b_col;
        } Circle;

        __kernel void circle_fractal(
            __global float* output,
            const int width, const int height,
            __global const Circle* circles,
            const int num_circles,
            const float limit)
        {
            int px = get_global_id(0);
            int py = get_global_id(1);
            if (px >= width || py >= height) return;

            float x = -limit + 2.0f * limit * px / (float)(width - 1);
            float y = -limit + 2.0f * limit * py / (float)(height - 1);

            int best_idx = -1;
            float best_r = 1e9f;
            for (int i = 0; i < num_circles; i++) {
                Circle c = circles[i];
                float dx = x - c.cx;
                float dy = y - c.cy;
                if (dx*dx + dy*dy <= c.r * c.r) {
                    if (c.r < best_r) {
                        best_r = c.r;
                        best_idx = i;
                    }
                }
            }

            int idx = (py * width + px) * 3;
            if (best_idx >= 0) {
                Circle c = circles[best_idx];
                output[idx]   = c.r_col;
                output[idx+1] = c.g_col;
                output[idx+2] = c.b_col;
            } else {
                output[idx] = 0.0f; output[idx+1] = 0.0f; output[idx+2] = 0.0f;
            }
        }
        """
        self.prg = cl.Program(self.ctx, kernel_src).build()
        self.knl = self.prg.circle_fractal

    def generate_circles_with_colors(self, base_radius, scale_factor,
                                     initial_children, recursive_children,
                                     max_level, min_size_factor, main_color):
        circles = []
        min_r = base_radius * min_size_factor

        def add_circle(cx, cy, r, color):
            circles.append((cx, cy, r, color[0], color[1], color[2]))

        add_circle(0.0, 0.0, base_radius, main_color)

        def recurse(cx, cy, radius, level):
            if level > max_level:
                return
            child_radius = radius * scale_factor
            if child_radius < min_r:
                return
            distance = radius - child_radius
            n = initial_children if level == 1 else recursive_children
            for i in range(n):
                angle = 2 * np.pi * i / n
                ix = cx + distance * np.cos(angle)
                iy = cy + distance * np.sin(angle)
                child_color = random_soft_color()
                add_circle(ix, iy, child_radius, child_color)
                recurse(ix, iy, child_radius, level + 1)

        if max_level >= 1:
            recurse(0.0, 0.0, base_radius, 1)
        return circles

    def render_gpu(self, circles, width, height, limit):
        circles_arr = np.array(circles, dtype=np.float32)
        circles_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                hostbuf=circles_arr)
        output = np.zeros((height, width, 3), dtype=np.float32)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)

        self.knl(self.queue, (width, height), None,
                 out_buf,
                 np.int32(width), np.int32(height),
                 circles_buf,
                 np.int32(len(circles)),
                 np.float32(limit))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

    def kececifractals_circle_gpu(self,
                              initial_children: int = 3,
                              recursive_children: int = 3,
                              text: str = "Keçeci Fractals",
                              font_size: int = 14,
                              font_color: str = "black",
                              font_style: str = "bold",
                              font_family: str = "Arial",
                              max_level: int = 4,
                              min_size_factor: float = 0.001,
                              scale_factor: float = 0.5,
                              base_radius: float = 12.0,
                              background_color=None,
                              initial_circle_color=None,
                              output_mode: str = "show",
                              filename: str = "kececi_fractal_circle_gpu",
                              dpi: int = 300,
                              width: int = 1024,
                              height: int = 1024,
                              view_limit: float = None):   # <-- YENİ
        if not isinstance(max_level, int) or max_level < 0:
            print("Error: max_level must be a non-negative integer.", file=sys.stderr)
            return
        if not (0 < scale_factor < 1):
            print("Error: scale_factor must be between 0 and 1.", file=sys.stderr)
            return

        # Ana daire rengi
        from matplotlib.colors import to_rgb
        if initial_circle_color is not None:
            main_color = to_rgb(initial_circle_color) if isinstance(initial_circle_color, str) else initial_circle_color
        else:
            main_color = random_soft_color()
    
        # Daireleri oluştur
        circles = self.generate_circles_with_colors(
            base_radius, scale_factor, initial_children, recursive_children,
            max_level, min_size_factor, main_color
        )
    
        # --- LİMİT HESAPLAMA ---
        if view_limit is not None:
            limit = view_limit
        else:
            limit = base_radius + 1.0
            if text and isinstance(text, str) and len(text) > 0:
                text_radius = base_radius + 0.8
                limit = max(limit, text_radius + font_size * 0.1)
            if circles:
                circles_arr = np.array(circles, dtype=np.float32)
                max_ext = circles_arr[:, 0:2].max() + circles_arr[:, 2].max()
                limit = max(limit, max_ext * 1.05)
    
        img = self.render_gpu(circles, width, height, limit)

        # Arka plan rengini uygula
        if background_color:
            bg_rgb = to_rgb(background_color) if isinstance(background_color, str) else background_color
            mask = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
            img[mask] = bg_rgb

        # Çizim
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[-limit, limit, -limit, limit], origin='lower')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axis('off')

        # Metin
        if text and isinstance(text, str) and len(text) > 0:
            text_radius = base_radius + 0.8
            fc = to_rgb(font_color) if isinstance(font_color, str) else (0,0,0)
            for i, char in enumerate(text):
                angle_deg = (360 / len(text) * i) - 90
                angle_rad = np.deg2rad(angle_deg)
                x_text = text_radius * np.cos(angle_rad)
                y_text = text_radius * np.sin(angle_rad)
                ax.text(x_text, y_text, char,
                        fontsize=font_size, ha="center", va="center",
                        color=fc, fontweight=font_style, fontfamily=font_family,
                        rotation=angle_deg + 90)

        plot_title = f"Keçeci Fractals GPU ({text})" if text else "Keçeci Circle Fractal GPU"
        plt.title(plot_title, fontsize=16)

        output_mode = output_mode.lower().strip()
        if output_mode == "show":
            plt.show()
        elif output_mode in ["png", "jpg", "jpeg", "svg"]:
            output_filename = f"{filename}.{output_mode}"
            plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
            print(f"Fractal saved to: {output_filename}")
            plt.close()
        else:
            print(f"Invalid output_mode: {output_mode}")
            plt.close()
"""
## Kullanım
kf_gpu = KececiFractalGPU()

kf_gpu.kececifractals_circle_gpu(
    text="Keçeci Fractals with GPU",
    max_level=3,
    background_color="#9a0a1a",
    output_mode="show",
    width=800, height=800,
    base_radius=12.0,      # büyük ana daire
    view_limit=14.0         # görüntü sınırını sabitle → daire ekranı doldurur
)
"""

class KececiFractalOpenGL:
    def __init__(self):
        # EGL bağlamını oluştur
        self._setup_egl()
        self._compile_shader()
        self.max_circles = 0
        self.circles_ssbo = None
        self.output_ssbo = None
        self.max_output_size = 0

    def _setup_egl(self):
        """EGL ile headless OpenGL bağlamı oluşturur."""
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if display == EGL_NO_DISPLAY:
            raise RuntimeError("EGL display alınamadı.")

        if not eglInitialize(display, None, None):
            raise RuntimeError("EGL başlatılamadı.")

        if not eglBindAPI(EGL_OPENGL_API):
            raise RuntimeError("EGL'ye OpenGL API bağlanamadı.")

        # Yapılandırma özellikleri
        config_attribs = [
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_STENCIL_SIZE, 8,
            EGL_NONE
        ]
        config_attribs = (ctypes.c_int * len(config_attribs))(*config_attribs)
        num_configs = ctypes.c_int()
        configs = (ctypes.c_void_p * 1)()

        if not eglChooseConfig(display, config_attribs, configs, 1, num_configs):
            raise RuntimeError("EGL yapılandırması seçilemedi.")
        config = configs[0]

        # PBuffer yüzeyi
        pbuffer_attribs = [
            EGL_WIDTH, 1,
            EGL_HEIGHT, 1,
            EGL_NONE
        ]
        pbuffer_attribs = (ctypes.c_int * len(pbuffer_attribs))(*pbuffer_attribs)
        surface = eglCreatePbufferSurface(display, config, pbuffer_attribs)
        if surface == EGL_NO_SURFACE:
            raise RuntimeError("EGL Pbuffer yüzeyi oluşturulamadı.")

        # Bağlam oluştur
        context_attribs = [
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_NONE
        ]
        context_attribs = (ctypes.c_int * len(context_attribs))(*context_attribs)
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs)
        if context == EGL_NO_CONTEXT:
            raise RuntimeError("EGL bağlamı oluşturulamadı.")

        # Aktif yap
        if not eglMakeCurrent(display, surface, surface, context):
            raise RuntimeError("EGL bağlamı aktif yapılamadı.")

        # Nesne referanslarını sakla
        self.egl_display = display
        self.egl_surface = surface
        self.egl_context = context

    def _compile_shader(self):
        compute_src = """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;

        struct Circle {
            float cx, cy, r;
            float r_col, g_col, b_col;
        };

        layout(std430, binding = 0) buffer CircleBuffer {
            Circle circles[];
        };

        layout(std430, binding = 1) buffer OutputBuffer {
            float outBuffer[];
        };

        uniform int u_width;
        uniform int u_height;
        uniform int u_num_circles;
        uniform float u_limit;

        void main() {
            uint px = gl_GlobalInvocationID.x;
            uint py = gl_GlobalInvocationID.y;
            if (px >= u_width || py >= u_height) return;

            float x = -u_limit + 2.0f * u_limit * float(px) / float(u_width - 1);
            float y = -u_limit + 2.0f * u_limit * float(py) / float(u_height - 1);

            int best_idx = -1;
            float best_r = 1e9f;
            for (int i = 0; i < u_num_circles; i++) {
                Circle c = circles[i];
                float dx = x - c.cx;
                float dy = y - c.cy;
                if (dx*dx + dy*dy <= c.r * c.r) {
                    if (c.r < best_r) {
                        best_r = c.r;
                        best_idx = i;
                    }
                }
            }

            uint idx = (py * u_width + px) * 3;
            if (best_idx >= 0) {
                Circle c = circles[best_idx];
                outBuffer[idx]   = c.r_col;
                outBuffer[idx+1] = c.g_col;
                outBuffer[idx+2] = c.b_col;
            } else {
                outBuffer[idx]   = 0.0;
                outBuffer[idx+1] = 0.0;
                outBuffer[idx+2] = 0.0;
            }
        }
        """
        shader = shaders.compileShader(compute_src, GL.GL_COMPUTE_SHADER)
        self.program = shaders.compileProgram(shader)
        self.loc_width = GL.glGetUniformLocation(self.program, "u_width")
        self.loc_height = GL.glGetUniformLocation(self.program, "u_height")
        self.loc_num = GL.glGetUniformLocation(self.program, "u_num_circles")
        self.loc_limit = GL.glGetUniformLocation(self.program, "u_limit")

    def _ensure_buffers(self, num_circles, output_floats):
        if self.circles_ssbo is None or num_circles > self.max_circles:
            if self.circles_ssbo:
                GL.glDeleteBuffers(1, [self.circles_ssbo])
            self.max_circles = max(num_circles, 1024)
            self.circles_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.circles_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,
                            self.max_circles * 6 * 4, None, GL.GL_DYNAMIC_DRAW)

        if self.output_ssbo is None or output_floats > self.max_output_size:
            if self.output_ssbo:
                GL.glDeleteBuffers(1, [self.output_ssbo])
            self.max_output_size = output_floats
            self.output_ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.output_ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,
                            output_floats * 4, None, GL.GL_DYNAMIC_READ)

    def generate_circles_with_colors(self, base_radius, scale_factor,
                                     initial_children, recursive_children,
                                     max_level, min_size_factor, main_color):
        circles = []
        min_r = base_radius * min_size_factor

        def add_circle(cx, cy, r, color):
            circles.append((cx, cy, r, color[0], color[1], color[2]))

        add_circle(0.0, 0.0, base_radius, main_color)

        def recurse(cx, cy, radius, level):
            if level > max_level:
                return
            child_radius = radius * scale_factor
            if child_radius < min_r:
                return
            distance = radius - child_radius
            n = initial_children if level == 1 else recursive_children
            for i in range(n):
                angle = 2 * np.pi * i / n
                ix = cx + distance * np.cos(angle)
                iy = cy + distance * np.sin(angle)
                child_color = random_soft_color()
                add_circle(ix, iy, child_radius, child_color)
                recurse(ix, iy, child_radius, level + 1)

        if max_level >= 1:
            recurse(0.0, 0.0, base_radius, 1)
        return circles

    def render_opengl(self, circles, width, height, limit):
        num_circles = len(circles)
        output_floats = width * height * 3
        self._ensure_buffers(num_circles, output_floats)

        circles_arr = np.array(circles, dtype=np.float32).flatten()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.circles_ssbo)
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, circles_arr.nbytes, circles_arr)

        GL.glUseProgram(self.program)
        GL.glUniform1i(self.loc_width, width)
        GL.glUniform1i(self.loc_height, height)
        GL.glUniform1i(self.loc_num, num_circles)
        GL.glUniform1f(self.loc_limit, limit)

        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.circles_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self.output_ssbo)

        gx = (width + 15) // 16
        gy = (height + 15) // 16
        GL.glDispatchCompute(gx, gy, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glFinish()

        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.output_ssbo)
        output_raw = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, output_floats * 4)
        output = np.frombuffer(output_raw, dtype=np.float32).reshape((height, width, 3))
        return output

    def kececifractals_circle_opengl(self,
                                     initial_children: int = 4,
                                     recursive_children: int = 4,
                                     text: str = "Keçeci Fractals",
                                     font_size: int = 14,
                                     font_color: str = "black",
                                     font_style: str = "bold",
                                     font_family: str = "Arial",
                                     max_level: int = 4,
                                     min_size_factor: float = 0.001,
                                     scale_factor: float = 0.5,
                                     base_radius: float = 12.0,
                                     background_color=None,
                                     initial_circle_color=None,
                                     output_mode: str = "show",
                                     filename: str = "kececi_fractal_circle_opengl",
                                     dpi: int = 300,
                                     width: int = 1024,
                                     height: int = 1024,
                                     view_limit: float = None):
        if not isinstance(max_level, int) or max_level < 0:
            print("Error: max_level must be a non-negative integer.", file=sys.stderr)
            return
        if not (0 < scale_factor < 1):
            print("Error: scale_factor must be between 0 and 1.", file=sys.stderr)
            return

        from matplotlib.colors import to_rgb
        if initial_circle_color is not None:
            main_color = to_rgb(initial_circle_color) if isinstance(initial_circle_color, str) else initial_circle_color
        else:
            main_color = random_soft_color()

        circles = self.generate_circles_with_colors(
            base_radius, scale_factor, initial_children, recursive_children,
            max_level, min_size_factor, main_color
        )

        if view_limit is not None:
            limit = view_limit
        else:
            limit = base_radius + 1.0
            if text and isinstance(text, str) and len(text) > 0:
                text_radius = base_radius + 0.8
                limit = max(limit, text_radius + font_size * 0.1)
            if circles:
                circles_arr = np.array(circles, dtype=np.float32)
                max_ext = circles_arr[:, 0:2].max() + circles_arr[:, 2].max()
                limit = max(limit, max_ext * 1.05)

        img = self.render_opengl(circles, width, height, limit)

        if background_color:
            bg_rgb = to_rgb(background_color) if isinstance(background_color, str) else background_color
            mask = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
            img[mask] = bg_rgb

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[-limit, limit, -limit, limit], origin='lower')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axis('off')

        if text and isinstance(text, str) and len(text) > 0:
            text_radius = base_radius + 0.8
            fc = to_rgb(font_color) if isinstance(font_color, str) else (0,0,0)
            for i, char in enumerate(text):
                angle_deg = (360 / len(text) * i) - 90
                angle_rad = np.deg2rad(angle_deg)
                x_text = text_radius * np.cos(angle_rad)
                y_text = text_radius * np.sin(angle_rad)
                ax.text(x_text, y_text, char,
                        fontsize=font_size, ha="center", va="center",
                        color=fc, fontweight=font_style, fontfamily=font_family,
                        rotation=angle_deg + 90)

        plot_title = f"Keçeci Fractals OpenGL ({text})" if text else "Keçeci Circle Fractal OpenGL"
        plt.title(plot_title, fontsize=16)

        output_mode = output_mode.lower().strip()
        if output_mode == "show":
            plt.show()
        elif output_mode in ["png", "jpg", "jpeg", "svg"]:
            output_filename = f"{filename}.{output_mode}"
            plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
            print(f"Fractal saved to: {output_filename}")
            plt.close()
        else:
            print(f"Invalid output_mode: {output_mode}")
            plt.close()

    def __del__(self):
        if hasattr(self, 'egl_display'):
            eglDestroyContext(self.egl_display, self.egl_context)
            eglDestroySurface(self.egl_display, self.egl_surface)
            eglTerminate(self.egl_display)

"""
#%matplotlib inline

ogl = KececiFractalOpenGL()
ogl.kececifractals_circle_opengl(
    text="OpenGL EGL",
    max_level=5,
    background_color="#8a0a1a",
    output_mode="show",
    width=800, height=800,
    base_radius=12.0,
    view_limit=14.0
)
"""

class KececiFractalVulkan:
    def __init__(self):
        self._setup_vulkan()
        self.max_circles = 0
        self.circles_buf = None
        self.circles_mem = None
        self.output_buf = None
        self.output_mem = None
        self.max_output_floats = 0

    def _setup_vulkan(self):
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="KececiFractal",
            applicationVersion=vk.VK_MAKE_VERSION(1,0,0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        self.instance = vk.vkCreateInstance(instance_info, None)

        phys_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not phys_devices:
            raise RuntimeError("Vulkan uyumlu cihaz bulunamadı.")
        self.phys_device = phys_devices[0]

        queue_family_index = 0
        queue_priority = 1.0
        device_queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[queue_priority],
        )
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[device_queue_create_info],
        )
        self.device = vk.vkCreateDevice(self.phys_device, device_create_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, queue_family_index, 0)

        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
        )
        self.cmd_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

        # Descriptor set layout
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=2, pBindings=bindings,
        )
        self.desc_set_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

        # Push constant: 4 adet int32 (hizalama için hepsi int)
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=16,
        )
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1, pSetLayouts=[self.desc_set_layout],
            pushConstantRangeCount=1, pPushConstantRanges=[push_range],
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipeline_layout_info, None)

        # Shader – limit'i float olarak alıp int push constant'tan dönüştürüyoruz
        glsl_src = """
        #version 450
        layout(local_size_x = 16, local_size_y = 16) in;

        struct Circle {
            float cx, cy, r;
            float r_col, g_col, b_col;
        };

        layout(std430, binding = 0) buffer CircleBuffer { Circle circles[]; };
        layout(std430, binding = 1) buffer OutputBuffer { float outBuffer[]; };

        layout(push_constant) uniform PushConstants {
            int width;
            int height;
            int num_circles;
            int limit_int;         // float yerine int gönderiyoruz
        } pc;

        void main() {
            uint px = gl_GlobalInvocationID.x;
            uint py = gl_GlobalInvocationID.y;
            if (px >= pc.width || py >= pc.height) return;

            float limit = float(pc.limit_int);
            float x = -limit + 2.0f * limit * float(px) / float(pc.width - 1);
            float y = -limit + 2.0f * limit * float(py) / float(pc.height - 1);

            int best_idx = -1;
            float best_r = 1e9f;
            for (int i = 0; i < pc.num_circles; i++) {
                Circle c = circles[i];
                float dx = x - c.cx;
                float dy = y - c.cy;
                if (dx*dx + dy*dy <= c.r * c.r) {
                    if (c.r < best_r) {
                        best_r = c.r;
                        best_idx = i;
                    }
                }
            }
            uint idx = (py * pc.width + px) * 3;
            if (best_idx >= 0) {
                Circle c = circles[best_idx];
                outBuffer[idx]   = c.r_col;
                outBuffer[idx+1] = c.g_col;
                outBuffer[idx+2] = c.b_col;
            } else {
                outBuffer[idx]   = 0.0; outBuffer[idx+1]=0.0; outBuffer[idx+2]=0.0;
            }
        }
        """
        spirv = self._compile_to_spirv(glsl_src)
        shader_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spirv), pCode=spirv,
        )
        self.shader_module = vk.vkCreateShaderModule(self.device, shader_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=self.shader_module, pName="main",
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info, layout=self.pipeline_layout,
        )
        self.pipeline = vk.vkCreateComputePipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None,
        )[0]

        # Descriptor pool
        pool_size = vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2)
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size],
        )
        self.desc_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.desc_pool, descriptorSetCount=1,
            pSetLayouts=[self.desc_set_layout],
        )
        self.desc_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

    def _compile_to_spirv(self, glsl_source):
        with tempfile.NamedTemporaryFile(suffix=".comp", delete=False) as f:
            f.write(glsl_source.encode())
            tmp_in = f.name
        tmp_out = tmp_in + ".spv"
        try:
            subprocess.run(
                ["glslangValidator", "-V", tmp_in, "-o", tmp_out],
                check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(tmp_out, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_in)
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)

    def _find_memory_type(self, type_filter, properties):
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.phys_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        return None

    def _create_buffer(self, size, usage, mem_properties):
        buf_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self.device, buf_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buf)
        mem_type = self._find_memory_type(mem_reqs.memoryTypeBits, mem_properties)
        if mem_type is None:
            raise RuntimeError("Uygun bellek türü bulunamadı")
        alloc = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size, memoryTypeIndex=mem_type,
        )
        mem = vk.vkAllocateMemory(self.device, alloc, None)
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        return buf, mem

    def _ensure_buffers(self, num_circles, output_floats):
        if self.circles_buf is None or num_circles > self.max_circles:
            if self.circles_buf:
                vk.vkDestroyBuffer(self.device, self.circles_buf, None)
                vk.vkFreeMemory(self.device, self.circles_mem, None)
            self.max_circles = max(num_circles, 1024)
            self.circles_buf, self.circles_mem = self._create_buffer(
                self.max_circles * 6 * 4,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )
        if self.output_buf is None or output_floats > self.max_output_floats:
            if self.output_buf:
                vk.vkDestroyBuffer(self.device, self.output_buf, None)
                vk.vkFreeMemory(self.device, self.output_mem, None)
            self.max_output_floats = output_floats
            self.output_buf, self.output_mem = self._create_buffer(
                output_floats * 4,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )

        # Descriptor set'i güncelle
        circle_desc = vk.VkDescriptorBufferInfo(buffer=self.circles_buf, offset=0, range=num_circles*6*4)
        out_desc = vk.VkDescriptorBufferInfo(buffer=self.output_buf, offset=0, range=output_floats*4)
        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.desc_set, dstBinding=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=circle_desc,
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.desc_set, dstBinding=1, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=out_desc,
            ),
        ]
        vk.vkUpdateDescriptorSets(self.device, 2, writes, 0, None)

    def generate_circles_with_colors(self, base_radius, scale_factor,
                                     initial_children, recursive_children,
                                     max_level, min_size_factor, main_color):
        circles = []
        min_r = base_radius * min_size_factor
        def add_circle(cx, cy, r, color):
            circles.append((cx, cy, r, color[0], color[1], color[2]))
        add_circle(0.0, 0.0, base_radius, main_color)

        def recurse(cx, cy, radius, level):
            if level > max_level:
                return
            child_radius = radius * scale_factor
            if child_radius < min_r:
                return
            distance = radius - child_radius
            n = initial_children if level == 1 else recursive_children
            for i in range(n):
                angle = 2 * np.pi * i / n
                ix = cx + distance * np.cos(angle)
                iy = cy + distance * np.sin(angle)
                child_color = random_soft_color()
                add_circle(ix, iy, child_radius, child_color)
                recurse(ix, iy, child_radius, level + 1)

        if max_level >= 1:
            recurse(0.0, 0.0, base_radius, 1)
        return circles

    def render_vulkan(self, circles, width, height, limit):
        num_circles = len(circles)
        output_floats = width * height * 3
        self._ensure_buffers(num_circles, output_floats)

        # Daire verisini yaz
        circles_arr = np.array(circles, dtype=np.float32).flatten()
        mapped = vk.vkMapMemory(self.device, self.circles_mem, 0, circles_arr.nbytes, 0)
        ffi.memmove(mapped, circles_arr.tobytes(), circles_arr.nbytes)
        vk.vkUnmapMemory(self.device, self.circles_mem)

        # Komut tamponu
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.cmd_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

        begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
        vk.vkBeginCommandBuffer(cmd, begin_info)

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                   0, 1, [self.desc_set], 0, None)

        # Push constants – 4 int
        push_data = np.array([width, height, num_circles, int(limit)], dtype=np.int32)
        push_ptr = ffi.new("int[]", push_data.tolist())
        vk.vkCmdPushConstants(cmd, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                              0, push_data.nbytes, push_ptr)

        vk.vkCmdDispatch(cmd, (width + 15) // 16, (height + 15) // 16, 1)
        vk.vkEndCommandBuffer(cmd)

        submit = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                 commandBufferCount=1, pCommandBuffers=[cmd])
        vk.vkQueueSubmit(self.queue, 1, submit, vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, [cmd])

        # Çıktıyı oku ve kopyala
        mapped = vk.vkMapMemory(self.device, self.output_mem, 0, output_floats * 4, 0)
        raw = bytes(mapped[:output_floats * 4])
        img = np.frombuffer(raw, dtype=np.float32).reshape((height, width, 3)).copy()
        vk.vkUnmapMemory(self.device, self.output_mem)
        return img

    # ---------- test (tamamen kırmızı) ----------
    def render_test_red(self, width=512, height=512):
        """Bütün pikselleri kırmızı yaparak Vulkan boru hattının çalıştığını test eder."""
        output_floats = width * height * 3
        self._ensure_buffers(1, output_floats)  # en az 1 daire için buffer ayır

        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.cmd_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

        begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
        vk.vkBeginCommandBuffer(cmd, begin_info)

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                   0, 1, [self.desc_set], 0, None)

        push_data = np.array([width, height, 0, 1], dtype=np.int32)  # num_circles=0, limit=1
        push_ptr = ffi.new("int[]", push_data.tolist())
        vk.vkCmdPushConstants(cmd, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                              0, push_data.nbytes, push_ptr)

        vk.vkCmdDispatch(cmd, (width + 15) // 16, (height + 15) // 16, 1)
        vk.vkEndCommandBuffer(cmd)

        submit = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                 commandBufferCount=1, pCommandBuffers=[cmd])
        vk.vkQueueSubmit(self.queue, 1, submit, vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, [cmd])

        mapped = vk.vkMapMemory(self.device, self.output_mem, 0, output_floats * 4, 0)
        raw = bytes(mapped[:output_floats * 4])
        img = np.frombuffer(raw, dtype=np.float32).reshape((height, width, 3)).copy()
        vk.vkUnmapMemory(self.device, self.output_mem)
        return img

    def kececifractals_circle_vulkan(self,
                                     initial_children=5, recursive_children=5,
                                     text="Keçeci Fractals", font_size=14,
                                     font_color="black", font_style="bold",
                                     font_family="Arial", max_level=4,
                                     min_size_factor=0.001, scale_factor=0.5,
                                     base_radius=12.0, background_color=None,
                                     initial_circle_color=None,
                                     output_mode="show",
                                     filename="kececi_fractal_circle_vulkan",
                                     dpi=300, width=1024, height=1024,
                                     view_limit=None):
        if not isinstance(max_level, int) or max_level < 0:
            print("Error: max_level must be a non-negative integer.", file=sys.stderr)
            return
        if not (0 < scale_factor < 1):
            print("Error: scale_factor must be between 0 and 1.", file=sys.stderr)
            return

        from matplotlib.colors import to_rgb
        if initial_circle_color:
            main_color = to_rgb(initial_circle_color) if isinstance(initial_circle_color, str) else initial_circle_color
        else:
            main_color = random_soft_color()

        circles = self.generate_circles_with_colors(
            base_radius, scale_factor, initial_children, recursive_children,
            max_level, min_size_factor, main_color,
        )

        if view_limit is not None:
            limit = view_limit
        else:
            limit = base_radius + 1.0
            if text:
                text_radius = base_radius + 0.8
                limit = max(limit, text_radius + font_size * 0.1)
            if circles:
                arr = np.array(circles, dtype=np.float32)
                max_ext = arr[:, 0:2].max() + arr[:, 2].max()
                limit = max(limit, max_ext * 1.05)

        img = self.render_vulkan(circles, width, height, limit)

        if background_color:
            bg = to_rgb(background_color) if isinstance(background_color, str) else background_color
            mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
            img[mask] = bg

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[-limit, limit, -limit, limit], origin='lower')
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect('equal'); ax.axis('off')

        if text:
            text_radius = base_radius + 0.8
            fc = to_rgb(font_color) if isinstance(font_color, str) else (0, 0, 0)
            for i, ch in enumerate(text):
                angle = (360 / len(text) * i) - 90
                rad = np.deg2rad(angle)
                x = text_radius * np.cos(rad)
                y = text_radius * np.sin(rad)
                ax.text(x, y, ch, fontsize=font_size, ha='center', va='center',
                        color=fc, fontweight=font_style, fontfamily=font_family,
                        rotation=angle + 90)

        title = f"Keçeci Fractals Vulkan ({text})" if text else "Keçeci Circle Fractal Vulkan"
        plt.title(title, fontsize=16)

        if output_mode == "show":
            plt.show()
        elif output_mode in ["png", "jpg", "jpeg", "svg"]:
            fname = f"{filename}.{output_mode}"
            plt.savefig(fname, dpi=dpi, bbox_inches='tight')
            print(f"Fractal saved to: {fname}")
            plt.close()
        else:
            print(f"Invalid output_mode: {output_mode}")
            plt.close()

"""
vkf = KececiFractalVulkan()
red = vkf.render_test_red(512, 512)
plt.imshow(red)
plt.show()
"""
"""
vkf = KececiFractalVulkan()
vkf.kececifractals_circle_vulkan(
    text="Keçeci Fraktals with Vulkan",
    max_level=2,
    background_color="#9a9a1a",
    output_mode="show",
    width=800, height=800,
    base_radius=12.0,
    view_limit=14.0
)
"""

# -------------------- Otomatik Seçim Sınıfı --------------------
class KececiFractalAuto:
    def __init__(self, prefer=None):
        self.backend = None
        self.backend_name = None

        candidates = [prefer.lower()] if prefer else ["vulkan", "opencl", "opengl", "cpu"]

        for candidate in candidates:
            if candidate == "vulkan" and KececiFractalVulkan is not None:
                try:
                    self.backend = KececiFractalVulkan()
                    # küçük test
                    img = self.backend.render_test_red(64,64) if hasattr(self.backend, 'render_test_red') else None
                    if img is not None and img.shape == (64,64,3):
                        self.backend_name = "Vulkan"
                        break
                except: pass
            elif candidate == "opencl" and KececiFractalOpenCL is not None:
                try:
                    self.backend = KececiFractalOpenCL()
                    circles = self.backend.generate_circles_with_colors(1.0,0.5,3,3,0,0.001,(1,0,0))
                    img = self.backend.render_gpu(circles, 64,64, 2.0)
                    if img is not None and img.shape == (64,64,3):
                        self.backend_name = "OpenCL"
                        break
                except: pass
            elif candidate == "opengl" and KececiFractalOpenGL is not None:
                try:
                    self.backend = KececiFractalOpenGL()
                    circles = self.backend.generate_circles_with_colors(1.0,0.5,3,3,0,0.001,(1,0,0))
                    img = self.backend.render_opengl(circles, 64,64, 2.0)
                    if img is not None and img.shape == (64,64,3):
                        self.backend_name = "OpenGL (EGL)"
                        break
                except: pass
            elif candidate == "cpu":
                if kececifractals_circle is not None:
                    self.backend = None
                    self.backend_name = "CPU (kececifractals)"
                    break

        if self.backend_name is None:
            raise RuntimeError("Hiçbir uygun arka uç bulunamadı.")

    def show(self, **kwargs):
        """
        Kullanıcıdan gelen tüm parametreleri seçilen arka uca aktarır.
        CPU için özel olarak kececifractals_circle fonksiyonunu çağırır.
        """
        if self.backend_name.startswith("CPU"):
            kececifractals_circle(**kwargs)
        elif self.backend_name == "Vulkan":
            self.backend.kececifractals_circle_vulkan(**kwargs)
        elif self.backend_name == "OpenCL":
            self.backend.kececifractals_circle_gpu(**kwargs)
        elif self.backend_name.startswith("OpenGL"):
            self.backend.kececifractals_circle_opengl(**kwargs)
        else:
            raise RuntimeError("Bilinmeyen arka uç")
            method(**kwargs)

    @property
    def active_backend(self):
        return self.backend_name

"""
kf = KececiFractalAuto()   # otomatik en iyiyi seçer

kf.show(
    initial_children=5,
    recursive_children=2,
    text="Keçeci Fractals with GPU",
    max_level=5,
    background_color="#8a8a1a",
    output_mode="show",
    width=800, height=800,
    base_radius=12.0,
    view_limit=14.0
)
"""
# ==============================================================================
# PART 5: MODULE TESTS
# ==============================================================================

if __name__ == "__main__":
    # Get current script name safely
    script_name = (
        os.path.basename(sys.argv[0]) if len(sys.argv) > 0 else "kececifractals.py"
    )
    print(f"--- Running Test Cases for {script_name} ---")

    # --- General-Purpose Fractal Tests ---
    print("\n--- PART 1: General-Purpose Fractal Tests ---")
    print("\n[Test 1.1: Displaying fractal on screen (show)]")
    kececifractals_circle(
        initial_children=5,
        recursive_children=4,
        text="Keçeci Fractals",
        max_level=3,
        output_mode="show",
    )

    print("\n[Test 1.2: Saving fractal as PNG]")
    kececifractals_circle(
        initial_children=7,
        recursive_children=3,
        text="Test PNG Save",
        background_color="#101030",  # Now accepts hex strings!
        initial_circle_color="yellow",  # Now accepts color names!
        output_mode="png",
        filename="test_fractal_generic",
    )

    # --- QEC Visualization Tests ---
    print("\n--- PART 2: QEC Visualization Tests ---")
    print("\n[Test 2.1: Generating an error-free 7-qubit code...]")
    visualize_qec_fractal(
        physical_qubits_per_level=7,
        recursion_level=1,
        error_qubits=[],
        filename="QEC_Model_Test_No_Errors",
    )

    print("\n[Test 2.2: Generating a 7-qubit code with a single error...]")
    visualize_qec_fractal(
        physical_qubits_per_level=7,
        recursion_level=1,
        error_qubits=[[3]],
        filename="QEC_Model_Test_Single_Error",
    )

    print("\n[Test 2.3: Generating a 2-level code with a deep-level error...]")
    visualize_qec_fractal(
        physical_qubits_per_level=5,
        recursion_level=2,
        error_qubits=[[4, 1]],
        filename="QEC_Model_Test_Deep_Error",
    )

    # --- 3D Fractal Tests ---
    if HAS_3D:
        print("\n--- PART 3: 3D Keçeci Fractal Tests ---")
        print("\n[Test 3.1: Generating basic 3D fractal...]")
        kececifractals_3d(
            num_children=6,
            max_level=3,
            output_mode="png",
            filename="test_3d_fractal_basic",
        )

        print("\n[Test 3.2: Generating complex 3D fractal...]")
        kececifractals_3d(
            num_children=12,
            max_level=4,
            scale_factor=0.35,
            color_scheme="viridis",
            elev=25,
            azim=60,
            output_mode="png",
            filename="test_3d_fractal_complex",
        )
    else:
        print("\n--- PART 3: 3D Keçeci Fractal Tests (Skipped - 3D not available) ---")

    print("\n--- All Tests Completed ---")
