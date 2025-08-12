import subprocess
import sys
import traceback

# 1. Adım: graphillion bağımlılığını pip ile kur
print("--> Installing graphillion dependency via pip...")
try:
    # subprocess.check_call, komut başarısız olursa doğrudan hata verir
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'graphillion>=1.0'])
    print("--> graphillion successfully installed.")
except subprocess.CalledProcessError:
    print("!!! Failed to install graphillion.")
    sys.exit(1) # Hata koduyla çık

# 2. Adım: kececilayout paketini import etmeyi dene
print("\n--> Attempting to import kececilayout...")
try:
    import kececilayout
    print("--> SUCCESS: kececilayout was imported successfully.")
except Exception:
    print("\n!!! IMPORT FAILED, TRACEBACK: !!!")
    traceback.print_exc()
    sys.exit(1) # Hata koduyla çık

# Tüm adımlar başarılıysa, 0 (başarı) koduyla çık
sys.exit(0)
