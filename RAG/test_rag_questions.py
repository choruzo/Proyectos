#!/usr/bin/env python3
"""
Script de prueba para evaluar el sistema RAG
Ejecuta varias preguntas de prueba y registra resultados
"""

import sys
import os
import json
from datetime import datetime

# Activar entorno y configurar encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None

# Importar el sistema RAG
from RAG_improved_v2_2_BOOSTING import (
    get_vectorstore,
    load_documents_with_metadata,
    AdaptiveSemanticChunker,
    ImprovedHybridRetriever,
    SimpleQueryExpander,
    FastReranker,
    SimpleRelevanceChecker,
    build_enhanced_prompt,
    DOCS_DIR,
    MODEL_NAME
)

from langchain_ollama import ChatOllama

# Preguntas de prueba
TEST_QUERIES = [
    # Queries cortas (deberían activar query expansion y alpha bajo)
    "apagar vm",
    "migrar vm",
    "crear snapshot",

    # Queries medianas
    "cómo crear una máquina virtual en ESXi",
    "configuración de red en VMware",

    # Queries largas (deberían usar más peso en vector search)
    "cuál es el procedimiento completo para realizar una migración vMotion de una máquina virtual entre hosts ESXi",

    # Query que podría no tener respuesta
    "cómo instalar Docker en ESXi",

    # Query técnica específica
    "configuración de vSwitch y VLAN",
]

def run_test():
    """Ejecuta pruebas del sistema RAG"""

    print("=" * 80)
    print("PRUEBAS AUTOMATIZADAS DEL SISTEMA RAG v2.2")
    print("=" * 80)

    # Inicializar sistema
    print("\n[1] Inicializando sistema...")
    vectorstore = get_vectorstore()

    if not vectorstore:
        print("[ERROR] No se pudo cargar la base de datos")
        return

    print("[OK] Vector store cargado")

    # Cargar documentos
    print("\n[2] Cargando documentos...")
    all_docs = load_documents_with_metadata(DOCS_DIR)
    chunker = AdaptiveSemanticChunker(chunk_size=1200, chunk_overlap=250)
    chunks = chunker.split_documents(all_docs)
    print(f"[OK] {len(chunks)} chunks creados")

    # Inicializar componentes
    print("\n[3] Inicializando componentes...")
    hybrid_retriever = ImprovedHybridRetriever(vectorstore, chunks, base_alpha=0.5)
    query_expander = SimpleQueryExpander()
    reranker = FastReranker(internal_docs=hybrid_retriever.internal_docs)
    relevance_checker = SimpleRelevanceChecker()
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
    print("[OK] Componentes listos")

    # Ejecutar pruebas
    print("\n[4] Ejecutando pruebas...")
    print("=" * 80)

    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"PRUEBA {i}/{len(TEST_QUERIES)}: {query}")
        print(f"{'='*80}")

        try:
            # Query expansion
            query_expanded, was_expanded = query_expander.expand(query)

            if was_expanded:
                print(f"[✓] Query expandida: {len(query_expanded.split()) - len(query.split())} términos añadidos")
            else:
                print(f"[→] Query sin expansión (longitud suficiente)")

            # Retrieval
            print(f"[→] Ejecutando búsqueda híbrida...")
            retrieval_results = hybrid_retriever.retrieve(query_expanded, k=40)

            print(f"[✓] {len(retrieval_results)} resultados iniciales")

            # Reranking
            print(f"[→] Reranking...")
            reranked = reranker.rerank(query, retrieval_results, top_k=8)

            print(f"[✓] Top {len(reranked)} resultados después de reranking")

            # Construir contexto
            context_parts = []
            sources = []

            for doc, score in reranked:
                source = doc.metadata.get('source', 'desconocido')
                source_name = os.path.basename(source)
                if source_name not in sources:
                    sources.append(source_name)
                context_parts.append(doc.page_content)

            context = "\n\n".join(context_parts)

            # Relevance check
            is_relevant, relevance_msg, relevance_score = relevance_checker.check_relevance(query, context)

            print(f"[→] Relevancia: {relevance_msg} (score: {relevance_score:.2%})")

            result_data = {
                'query': query,
                'query_length': len(query.split()),
                'was_expanded': was_expanded,
                'num_initial_results': len(retrieval_results),
                'num_reranked': len(reranked),
                'is_relevant': is_relevant,
                'relevance_score': relevance_score,
                'sources': sources,
                'avg_rerank_score': sum(s for _, s in reranked) / len(reranked) if reranked else 0,
            }

            if is_relevant:
                # Generar respuesta
                print(f"[→] Generando respuesta LLM...")
                prompt = build_enhanced_prompt(query, context, sources)

                ai_message = llm.invoke(prompt)
                response = getattr(ai_message, 'content', str(ai_message))

                result_data['response'] = response[:200] + "..." if len(response) > 200 else response
                result_data['response_length'] = len(response)

                print(f"\n[RESPUESTA]:")
                print("-" * 80)
                print(response[:300] + "..." if len(response) > 300 else response)
                print("-" * 80)

            else:
                result_data['response'] = "No relevante - no se generó respuesta"
                result_data['response_length'] = 0
                print(f"[!] Contexto no relevante - saltando generación de respuesta")

            print(f"\n[MÉTRICAS]:")
            print(f"  - Fuentes: {', '.join(sources)}")
            print(f"  - Score promedio: {result_data['avg_rerank_score']:.3f}")
            print(f"  - Contexto relevante: {'Sí' if is_relevant else 'No'}")

            results.append(result_data)

        except Exception as e:
            print(f"[ERROR] Error en prueba: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'query': query,
                'error': str(e)
            })

    # Guardar resultados
    print("\n" + "=" * 80)
    print("RESUMEN DE PRUEBAS")
    print("=" * 80)

    successful = sum(1 for r in results if 'error' not in r and r.get('is_relevant'))
    not_relevant = sum(1 for r in results if 'error' not in r and not r.get('is_relevant'))
    errors = sum(1 for r in results if 'error' in r)
    expanded = sum(1 for r in results if r.get('was_expanded'))

    print(f"\nResultados:")
    print(f"  - Exitosas (relevantes): {successful}/{len(TEST_QUERIES)}")
    print(f"  - No relevantes: {not_relevant}/{len(TEST_QUERIES)}")
    print(f"  - Errores: {errors}/{len(TEST_QUERIES)}")
    print(f"  - Queries expandidas: {expanded}/{len(TEST_QUERIES)}")

    avg_relevance = sum(r.get('relevance_score', 0) for r in results if 'error' not in r) / len(results) if results else 0
    print(f"  - Relevancia promedio: {avg_relevance:.2%}")

    # Guardar JSON
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(TEST_QUERIES),
            'successful': successful,
            'not_relevant': not_relevant,
            'errors': errors,
            'expanded_queries': expanded,
            'avg_relevance': avg_relevance,
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Resultados guardados en: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
