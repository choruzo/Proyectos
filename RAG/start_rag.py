#!/usr/bin/env python3
"""
Script de Inicio Optimizado para RAG
Muestra progreso claro durante la carga inicial
"""

import sys
import time

def print_progress(message, end='\n'):
    """Print con flush inmediato para Windows"""
    print(message, end=end, flush=True)

def check_ollama():
    """Verifica que Ollama esté corriendo"""
    print_progress("\n[1/5] Verificando Ollama...")
    try:
        import ollama
        models = ollama.list()
        print_progress(f"  [OK] Ollama conectado - {len(models.get('models', []))} modelos disponibles")
        return True
    except Exception as e:
        print_progress(f"  [ERROR] Ollama no disponible: {e}")
        print_progress("\n  Asegúrate de que Ollama esté corriendo:")
        print_progress("    1. Abre Ollama desde el menú inicio")
        print_progress("    2. Verifica con: ollama list")
        return False

def check_dependencies():
    """Verifica dependencias críticas"""
    print_progress("\n[2/5] Verificando dependencias...")
    
    required = [
        'langchain',
        'langchain_ollama',
        'langchain_community',
        'langchain_text_splitters',
        'chromadb'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print_progress(f"  [OK] {pkg}")
        except ImportError:
            print_progress(f"  [FALTA] {pkg}")
            missing.append(pkg)
    
    if missing:
        print_progress(f"\n  Instala los paquetes faltantes:")
        print_progress(f"    pip install {' '.join(missing)}")
        return False
    
    return True

def check_docs_folder():
    """Verifica carpeta de documentos"""
    print_progress("\n[3/5] Verificando documentos...")
    import os
    
    if not os.path.exists('docs'):
        print_progress("  [!] Carpeta 'docs' no existe")
        print_progress("  Creando carpeta...")
        os.makedirs('docs', exist_ok=True)
        print_progress("  [OK] Carpeta creada - coloca tus documentos aquí")
        return False
    
    # Contar archivos
    files = []
    for root, _, filenames in os.walk('docs'):
        for f in filenames:
            if f.lower().endswith(('.pdf', '.md', '.markdown', '.txt')):
                files.append(os.path.join(root, f))
    
    if not files:
        print_progress("  [!] No hay documentos en 'docs/'")
        print_progress("  Coloca archivos .pdf, .md o .txt en la carpeta 'docs/'")
        return False
    
    print_progress(f"  [OK] {len(files)} documentos encontrados:")
    for f in files[:5]:  # Mostrar primeros 5
        print_progress(f"    - {os.path.basename(f)}")
    if len(files) > 5:
        print_progress(f"    ... y {len(files)-5} más")
    
    return True

def estimate_loading_time(num_files):
    """Estima tiempo de carga"""
    # Aproximadamente 2-5 segundos por archivo para embeddings
    min_time = num_files * 2
    max_time = num_files * 5
    return min_time, max_time

def check_existing_db():
    """Verifica si ya existe base de datos"""
    print_progress("\n[4/5] Verificando base de datos...")
    import os
    
    if os.path.exists('db_esxi') and os.path.exists('db_esxi/index_manifest.json'):
        print_progress("  [OK] Base de datos existente encontrada")
        print_progress("  No será necesario crear embeddings")
        return True
    else:
        print_progress("  [!] Base de datos no existe - se creará en el primer arranque")
        
        # Estimar tiempo
        import glob
        files = glob.glob('docs/**/*.*', recursive=True)
        num_files = len([f for f in files if f.lower().endswith(('.pdf', '.md', '.txt'))])
        
        if num_files > 0:
            min_t, max_t = estimate_loading_time(num_files)
            print_progress(f"  [INFO] Creación de embeddings tomará ~{min_t}-{max_t} segundos")
            print_progress(f"        ({num_files} archivos x ~2-5 seg/archivo)")
            print_progress("\n  ¡IMPORTANTE! El primer arranque será LENTO")
            print_progress("  Esto solo ocurre una vez. Arranques posteriores serán rápidos.")
        
        return False

def main():
    """Verificación pre-arranque"""
    print_progress("=" * 70)
    print_progress("  VERIFICACIÓN PREVIA AL ARRANQUE DEL SISTEMA RAG")
    print_progress("=" * 70)
    
    all_ok = True
    
    # Checks
    if not check_ollama():
        all_ok = False
    
    if not check_dependencies():
        all_ok = False
    
    has_docs = check_docs_folder()
    
    has_db = check_existing_db()
    
    # Resumen
    print_progress("\n[5/5] Resumen:")
    
    if all_ok and has_docs:
        print_progress("  [OK] Todos los requisitos están listos")
        
        if has_db:
            print_progress("\n  Tiempo estimado de arranque: ~2-5 segundos")
        else:
            print_progress("\n  [!] PRIMER ARRANQUE - Será lento (varios minutos)")
            print_progress("      El sistema creará embeddings para todos los documentos")
            print_progress("      Verás el progreso en tiempo real")
        
        print_progress("\n" + "=" * 70)
        print_progress("  ¿Listo para iniciar?")
        print_progress("=" * 70)
        
        response = input("\n  Presiona ENTER para continuar (o Ctrl+C para cancelar): ")
        
        print_progress("\n" + "=" * 70)
        print_progress("  INICIANDO SISTEMA RAG...")
        print_progress("=" * 70 + "\n")
        
        # Importar y ejecutar el sistema principal
        try:
            import RAG_improvedV1
            RAG_improvedV1.main()
        except KeyboardInterrupt:
            print_progress("\n\n[!] Arranque cancelado por el usuario")
            return 1
        except Exception as e:
            print_progress(f"\n[ERROR] Error al iniciar: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    else:
        print_progress("\n  [!] Hay problemas que resolver antes de iniciar")
        
        if not all_ok:
            print_progress("\n  Soluciona los errores indicados arriba")
        
        if not has_docs:
            print_progress("\n  1. Copia tus documentos (.pdf, .md, .txt) a la carpeta 'docs/'")
            print_progress("  2. Ejecuta este script nuevamente")
        
        print_progress("\n" + "=" * 70)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
