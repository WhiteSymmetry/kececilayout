# -*- coding: utf-8 -*-
import io
import re
from setuptools import setup, find_packages
import sys


# BU SATIRLAR SORUNUN KALICI ÇÖZÜMÜDÜR.
# Python'a, README.md dosyasını hangi işletim sisteminde olursa olsun
# her zaman UTF-8 kodlamasıyla okumasını söylüyoruz.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def get_version():
    with open('kececilayout/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="kececilayout",
    version=get_version(),
    description="A deterministic node placement algorithm used in graph visualization. In this layout, nodes are arranged sequentially along a defined primary axis. Each subsequent node is then alternately offset along a secondary, perpendicular axis, typically moving to one side of the primary axis and then the other. Often, the magnitude of this secondary offset increases as nodes progress along the primary axis, creating a characteristic zig-zag or serpentine pattern.",
    long_description=long_description,
    long_description_content_type="text/markdown", # Bu satır da önemlidir
    author="Mehmet Keçeci",
    maintainer="Mehmet Keçeci",
    author_email="bilginomi@yaani.com",
    maintainer_email="bilginomi@yaani.com",
    url="https://github.com/WhiteSymmetry/kececilayout",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "numpy",
        "matplotlib",
        "pycairo",
        "cairocffi"
    ],
    extras_require={
        "all": ["cairo", "python-louvain", "python-igraph", "networkit", "rustworkx", "graphillion"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.10',
    license="MIT",
)




