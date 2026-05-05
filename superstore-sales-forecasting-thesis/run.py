import subprocess
import os
import sys

# Obtener la carpeta donde está este script (run.py)
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_APP = os.path.join(CARPETA_ACTUAL, "app_dashboard.py")

def ejecutar_streamlit():
    """Ejecuta la app de Streamlit desde el botón de VS Code"""
    
    # Verificar si el archivo existe
    if not os.path.exists(ARCHIVO_APP):
        print(f"❌ Error: No se encuentra el archivo 'app_dashboard.py'")
        print(f"📁 Buscando en: {CARPETA_ACTUAL}")
        return
    
    print(f"✅ Archivo encontrado: {ARCHIVO_APP}")
    print(f"🚀 Ejecutando Streamlit...")
    print(f"💡 La app se abrirá en tu navegador")
    print(f"🛑 Presiona Ctrl+C en la terminal para detener la app\n")
    
    # Cambiar al directorio correcto
    os.chdir(CARPETA_ACTUAL)
    
    # Método 1: Usar python -m streamlit (más confiable)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_dashboard.py"])
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Soluciones:")
        print("1. Instala Streamlit: pip install streamlit")
        print("2. O usa: python -m pip install streamlit")

if __name__ == "__main__":
    ejecutar_streamlit()