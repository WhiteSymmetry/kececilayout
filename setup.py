from setuptools import setup, find_packages

setup(
    name="kececilayout",
    version="0.2.4",
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
        "rustworkx",
        "igraph",
        "networkit",
        "graphillion",
        "pycairo" ,
        "cairocffi",
    ],
    extras_require={

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    license="MIT",
)
