import subprocess
import sys
import importlib.util
import traceback
import os

def check_and_install_package(package_name, min_version=None):
    """Paketi kontrol et, kurulu değilse/versiyon eskiyse kur/güncelle"""
    # 1. Paket kurulu mu kontrol et
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"➤ {package_name} YOK - KURULUYOR...")
        install_cmd = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', package_name]
        if min_version:
            install_cmd.extend([f'>={min_version}'])
    else:
        print(f"➤ {package_name} KURULU - GÜNCELLENİYOR...")
        install_cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', package_name]
    
    try:
        subprocess.check_call(install_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ {package_name} başarıyla güncellendi!")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} kurulamadı!")
        return False

def main():
    print("🔧 KececiLayout Kurulum/Test Aracı")
    print("=" * 50)
    
    # 1. graphillion kontrol et ve kur/güncelle
    if not check_and_install_package('graphillion', '1.0'):
        print("❌ Graphillion kurulamadı! Çıkılıyor...")
        sys.exit(1)
    
    # 2. kececilayout kontrol et ve kur/güncelle
    print("\n➤ kececilayout paketi kontrol ediliyor...")
    if not check_and_install_package('kececilayout'):
        print("❌ kececilayout kurulamadı! Çıkılıyor...")
        sys.exit(1)
    
    # 3. Import test et (Jupyter/konsol uyumlu)
    print("\n🔍 Import testi yapılıyor...")
    try:
        import kececilayout as kl
        print("✅ IMPORT BAŞARILI!")
        
        # Test çizimi (Jupyter uyumlu)
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.gnp_random_graph(5, 0.3)
            pos = kl.kececi_layout_2d(G)
            plt.figure(figsize=(6,4))
            nx.draw(G, pos, with_labels=True)
            plt.title("✅ KececiLayout Test - BAŞARILI!")
            plt.show()
            print("🎉 Test grafiği başarıyla çizildi!")
            
        except ImportError:
            print("ℹ️  NetworkX/Matplotlib yok - sadece import testi yapıldı")
        
        # Başarı dosyası oluştur
        with open("kececi_test_success.log", "w", encoding="utf-8") as f:
            f.write("OK")
        print("✅ test_success.log oluşturuldu")
        return 0
        
    except Exception as e:
        print(f"\n❌ IMPORT HATASI: {e}")
        print("🔍 Detaylar test_error.log dosyasına kaydedildi")
        
        # Jupyter'de traceback göster, konsolda dosyaya yaz
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            traceback.print_exc()
        else:
            with open("test_error.log", "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
