import subprocess
import sys
import traceback

# 1. Adım: graphillion bağımlılığını pip ile kur
print("--> Installing graphillion dependency via pip...")
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'graphillion>=1.0'])
    print("--> graphillion successfully installed.")
except subprocess.CalledProcessError as e:
    with open("test_error.log", "w", encoding="utf-8") as f:
        f.write("Failed to install graphillion.\n")
        f.write(str(e))
    sys.exit(1)

# 2. Adım: kececilayout paketini import etmeyi dene
print("\n--> Attempting to import kececilayout...")
try:
    import kececilayout
    print("--> SUCCESS: kececilayout was imported successfully.")
    # Başarıyı belirtmek için boş bir dosya oluştur
    with open("test_success.log", "w", encoding="utf-8") as f:
        f.write("OK")
    sys.exit(0)
except Exception:
    # HATA DURUMUNDA: Traceback'i bir dosyaya yazdır
    print("\n!!! IMPORT FAILED, SAVING TRACEBACK TO test_error.log !!!")
    with open("test_error.log", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    sys.exit(1)
