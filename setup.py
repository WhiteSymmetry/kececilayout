import re
from setuptools import setup, find_packages

# __version__'ı __init__.py'den oku
def get_version():
    with open('kececilayout/__init__.py', 'r') as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="kececilayout",
    #version="0.2.7",
    version=get_version(),
    description="A deterministic node placement algorithm used in graph visualization. In this layout, nodes are arranged sequentially along a defined primary axis. Each subsequent node is then alternately offset along a secondary, perpendicular axis, typically moving to one side of the primary axis and then the other. Often, the magnitude of this secondary offset increases as nodes progress along the primary axis, creating a characteristic zig-zag or serpentine pattern.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mehmet Keçeci",
    maintainer="Mehmet Keçeci",
    author_email="bilginomi@yaani.com",
    maintainer_email="bilginomi@yaani.com",
    url="https://github.com/WhiteSymmetry/kececilayout",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "numpy",
        "pycairo",
        "cairocffi"
    ],
    extras_require={
        "all": ["igraph", "networkit", "rustworkx", "graphillion"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.9',
    license="MIT",
)


