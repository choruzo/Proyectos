#!/usr/bin/env python3
"""
Script de Verificación de Dependencias
Verifica que todas las dependencias necesarias estén instaladas correctamente
"""

import sys
from importlib import import_module

REQUIRED_PACKAGES = {
    'langchain': '1.2.7',
    'langchain_core': '1.2.7',
    'langchain_community': '0.4.1',
    'langchain_text_splitters': '1.1.0',
    'langchain_ollama': '1.0.1',
    'langchain_chroma': '1.1.0',
    'chromadb': '1.4.1',
    'ollama': '0.6.1',
    'pypdf': '6.6.2',
}

def check_package(package_name, expected_version=None):
    """Verifica si un paquete está instalado"""
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if expected_version and version != 'unknown':
            if version != expected_version:
                return 'warning', f"Versión {version} (esperada: {expected_version})"
        
        return 'ok', version
    except ImportError:
        return 'error', 'No instalado'
    except Exception as e:
        return 'error', str(e)

def main():
    print("=" * 70)
    print(" " * 15 + "VERIFICACIÓN DE DEPENDENCIAS")
    print("=" * 70)
    print()
    
    all_ok = True
    warnings = []
    errors = []
    
    for package, expected_version in REQUIRED_PACKAGES.items():
        status, info = check_package(package, expected_version)
        
        if status == 'ok':
            symbol = "✓"
            color = ""
        elif status == 'warning':
            symbol = "⚠"
            color = ""
            warnings.append(f"{package}: {info}")
        else:
            symbol = "✗"
            color = ""
            errors.append(f"{package}: {info}")
            all_ok = False
        
        print(f"{symbol} {package:<30} {info}")
    
    print()
    print("=" * 70)
    
    if errors:
        print("\n❌ ERRORES ENCONTRADOS:")
        for error in errors:
            print(f"  - {error}")
        print("\nPara instalar los paquetes faltantes:")
        print("  pip install -r requirements.txt --break-system-packages")
    
    if warnings:
        print("\n⚠️  ADVERTENCIAS:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nLas versiones diferentes pueden funcionar, pero no están garantizadas.")
    
    if all_ok and not warnings:
        print("\n✅ TODAS LAS DEPENDENCIAS ESTÁN CORRECTAMENTE INSTALADAS")
        print("\nVerificando componentes específicos...")
        
        # Verificar imports específicos
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
            print("  ✓ Text Splitters OK")
        except Exception as e:
            print(f"  ✗ Text Splitters ERROR: {e}")
            all_ok = False
        
        try:
            from langchain_ollama import OllamaEmbeddings, ChatOllama
            print("  ✓ Ollama Integration OK")
        except Exception as e:
            print(f"  ✗ Ollama Integration ERROR: {e}")
            all_ok = False
        
        try:
            from langchain_community.vectorstores import Chroma
            print("  ✓ ChromaDB OK")
        except Exception as e:
            print(f"  ✗ ChromaDB ERROR: {e}")
            all_ok = False
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            print("  ✓ PDF Loader OK")
        except Exception as e:
            print(f"  ✗ PDF Loader ERROR: {e}")
            all_ok = False
        
        try:
            from langchain_core.documents import Document
            print("  ✓ Document Core OK")
        except Exception as e:
            print(f"  ✗ Document Core ERROR: {e}")
            all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("\n🚀 ¡Sistema listo para ejecutar!")
        print("\nPuedes ejecutar:")
        print("  python RAG_improved.py")
        return 0
    else:
        print("\n⚠️  Hay problemas que deben resolverse antes de ejecutar el sistema.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
