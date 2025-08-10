# noxfile.py

import nox
import os

# Test edilecek Python versiyonlarını listele
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]

# Varsayılan olarak çalıştırılacak oturumları belirle
# (Terminalde sadece 'nox' yazıldığında bu oturumlar çalışır)
nox.options.sessions = ["lint", "tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """
    Run the test suite for all specified Python versions.
    
    Bu oturum 'nox -s tests-3.11' gibi komutlarla tetiklenir.
    """
    # 1. Gerekli bağımlılıkları kur
    #    'pytest', 'pytest-mock' ve projenin kendisini kur ('-e .' ile).
    session.install("-e", ".[test]") # pyproject.toml'daki [project.optional-dependencies] test grubunu kurar

    # 2. Pytest'i çalıştır
    #    --cov ile kod kapsamı (code coverage) raporu oluştur.
    session.run("pytest", "--cov=kececilayout", "--cov-report=xml")


@nox.session(python="3.11") # Linting genellikle tek bir versiyonda yapılır
def lint(session):
    """
    Run linters to check code style and quality.
    """
    # Linting araçlarını kur
    session.install("ruff")
    
    # Kodu kontrol et
    session.run("ruff", "check", "kececilayout", "tests")
    session.run("ruff", "format", "--check", "kececilayout", "tests")


# --- Proje Yapınıza Göre Eklemeler ---
# Eğer 'pyproject.toml' dosyanızda test bağımlılıkları tanımlı değilse,
# tests oturumundaki install satırını şu şekilde değiştirebilirsiniz:
#
# @nox.session(python=PYTHON_VERSIONS)
# def tests(session):
#     # Bağımlılıkları manuel olarak kur
#     session.install("pytest", "pytest-mock", "numpy", "networkx", "igraph", "rustworkx", "networkit", "graphillion", "matplotlib")
#     # Projeyi düzenlenebilir modda kur
#     session.install("-e", ".")
#     session.run("pytest", "--cov=kececilayout", "--cov-report=xml")
