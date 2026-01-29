import os
import glob
import shutil
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import TextLoader
except Exception:
    class TextLoader:
        def __init__(self, path, encoding='utf-8'):
            self.path = path
            self.encoding = encoding

        def load(self):
            with open(self.path, 'r', encoding=self.encoding) as f:
                text = f.read()
            return [Document(page_content=text, metadata={'source': self.path})]

# --- CONFIGURACIÓN ---
DB_DIR = "db_esxi"
MODEL_NAME = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
DOCS_DIR = "docs"
LOGS_DIR = "logs"

# Configurar logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'rag_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Métricas de calidad del retrieval"""
    query: str
    num_chunks_retrieved: int
    avg_similarity_score: float
    sources_used: List[str]
    context_length: int
    retrieval_method: str


class BM25Retriever:
    """Implementación simple de BM25 para búsqueda por keywords"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.avgdl = 0
        self.N = len(documents)
        self.idf = {}
        
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenización simple"""
        import re
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return [t for t in tokens if len(t) > 2]
    
    def _build_index(self):
        """Construir índice BM25"""
        logger.info("Construyendo índice BM25...")
        
        # Calcular frecuencias de términos por documento
        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))
            
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.doc_term_freqs.append(term_freq)
        
        # Calcular longitud promedio de documento
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calcular IDF
        df = {}
        for term_freq in self.doc_term_freqs:
            for term in term_freq.keys():
                df[term] = df.get(term, 0) + 1
        
        import math
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
        
        logger.info(f"Índice BM25 construido con {self.N} documentos")
    
    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Tuple[Document, float]]:
        """Buscar documentos relevantes usando BM25"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, (doc, term_freq, doc_len) in enumerate(zip(self.documents, self.doc_term_freqs, self.doc_lengths)):
            score = 0.0
            for token in query_tokens:
                if token in term_freq:
                    tf = term_freq[token]
                    idf = self.idf.get(token, 0)
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self.avgdl))
            
            scores.append((doc, score))
        
        # Ordenar por score descendente
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class SemanticChunker:
    """Chunking mejorado con respeto a límites semánticos"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separadores semánticos en orden de prioridad
        self.separators = [
            "\n\n\n",  # Múltiples líneas vacías
            "\n\n",    # Párrafos
            "\n",      # Líneas
            ". ",      # Frases
            "! ",
            "? ",
            "; ",
            ", ",
            " ",       # Palabras
            ""         # Caracteres
        ]
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split con respeto a límites semánticos"""
        chunks = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            file_type = Path(source).suffix.lower()
            
            # Estrategia específica por tipo de archivo
            if file_type in ['.md', '.markdown']:
                chunks.extend(self._split_markdown(doc))
            else:
                chunks.extend(self._split_recursive(doc))
        
        logger.info(f"Documentos divididos en {len(chunks)} chunks semánticos")
        return chunks
    
    def _split_markdown(self, doc: Document) -> List[Document]:
        """Split específico para Markdown preservando estructura"""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        try:
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_chunks = md_splitter.split_text(doc.page_content)
            
            # Añadir metadata de headers
            result = []
            for chunk in md_chunks:
                metadata = {**doc.metadata}
                metadata.update(chunk.metadata)
                result.append(Document(page_content=chunk.page_content, metadata=metadata))
            
            # Si los chunks son muy grandes, subdividir
            final_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators
            )
            
            for chunk in result:
                if len(chunk.page_content) > self.chunk_size:
                    sub_chunks = text_splitter.split_documents([chunk])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
            
        except Exception as e:
            logger.warning(f"Error en split de Markdown: {e}, usando split recursivo")
            return self._split_recursive(doc)
    
    def _split_recursive(self, doc: Document) -> List[Document]:
        """Split recursivo respetando límites semánticos"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        return text_splitter.split_documents([doc])


class HybridRetriever:
    """Retriever híbrido que combina búsqueda vectorial y BM25"""
    
    def __init__(self, vectorstore, documents: List[Document], alpha: float = 0.5):
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever(documents)
        self.alpha = alpha  # Peso para vectorial (1-alpha para BM25)
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieval híbrido con normalización de scores"""
        
        # Búsqueda vectorial con scores
        try:
            vector_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
            logger.info(f"Búsqueda vectorial: {len(vector_results)} resultados")
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            vector_results = [(doc, 0.0) for doc in self.vectorstore.similarity_search(query, k=k)]
        
        # Búsqueda BM25
        try:
            bm25_results = self.bm25.search(query, k=k)
            logger.info(f"Búsqueda BM25: {len(bm25_results)} resultados")
        except Exception as e:
            logger.error(f"Error en búsqueda BM25: {e}")
            bm25_results = []
        
        # Normalizar scores
        def normalize_scores(results):
            if not results:
                return []
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            range_score = max_score - min_score if max_score != min_score else 1.0
            
            return [(doc, (score - min_score) / range_score) for doc, score in results]
        
        vector_normalized = normalize_scores(vector_results)
        bm25_normalized = normalize_scores(bm25_results)
        
        # Combinar scores
        combined_scores = {}
        for doc, score in vector_normalized:
            doc_id = id(doc)
            combined_scores[doc_id] = {
                'doc': doc,
                'vector_score': score * self.alpha,
                'bm25_score': 0.0
            }
        
        for doc, score in bm25_normalized:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['bm25_score'] = score * (1 - self.alpha)
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'bm25_score': score * (1 - self.alpha)
                }
        
        # Calcular score final y ordenar
        final_results = []
        for doc_id, data in combined_scores.items():
            final_score = data['vector_score'] + data['bm25_score']
            final_results.append((data['doc'], final_score))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Retrieval híbrido: {len(final_results)} resultados combinados")
        return final_results[:k]


class RelevanceChecker:
    """Verifica si el contexto recuperado es relevante para la pregunta"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def check_relevance(self, query: str, context: str) -> Tuple[bool, str]:
        """Verifica relevancia del contexto"""
        
        # Verificación rápida por keywords
        query_tokens = set(query.lower().split())
        context_tokens = set(context.lower().split())
        overlap = len(query_tokens & context_tokens) / len(query_tokens) if query_tokens else 0
        
        if overlap < 0.1:  # Muy poco overlap
            return False, "El contexto no parece relacionado con la pregunta (bajo overlap de keywords)"
        
        # Si hay buen overlap, asumimos relevancia
        if overlap > 0.3:
            return True, "Contexto relevante"
        
        # Para casos intermedios, podríamos usar el LLM (costoso)
        # Por ahora, ser conservador y aceptar
        return True, "Contexto potencialmente relevante"


def compute_file_metadata(path: str) -> Dict:
    """Calcula metadata completa de un archivo"""
    try:
        st = os.stat(path)
        h = hashlib.sha1()
        with open(path, 'rb') as fb:
            while True:
                chunk = fb.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        
        return {
            'path': os.path.normpath(path),
            'mtime': int(st.st_mtime),
            'size': int(st.st_size),
            'sha1': h.hexdigest()
        }
    except Exception as e:
        logger.error(f"Error calculando metadata de {path}: {e}")
        return {
            'path': os.path.normpath(path),
            'mtime': None,
            'size': None,
            'sha1': None
        }


def load_documents_with_metadata(docs_dir: str) -> List[Document]:
    """Carga documentos con metadata enriquecida"""
    all_docs = []
    
    if not os.path.exists(docs_dir):
        logger.warning(f"Directorio {docs_dir} no existe")
        return all_docs
    
    pattern = os.path.join(docs_dir, "**", "*.*")
    files = glob.glob(pattern, recursive=True)
    
    supported = ('.pdf', '.md', '.markdown', '.txt')
    
    for file_path in files:
        if os.path.isdir(file_path):
            continue
        
        if any(part == DB_DIR for part in file_path.split(os.sep)):
            continue
        
        lower = file_path.lower()
        
        try:
            # Metadata base
            rel_path = os.path.relpath(file_path, docs_dir)
            path_parts = Path(rel_path).parts
            
            base_metadata = {
                'source': file_path,
                'filename': os.path.basename(file_path),
                'file_type': Path(file_path).suffix,
                'directory': os.path.dirname(rel_path),
                'depth': len(path_parts) - 1,
            }
            
            # Cargar según tipo
            if lower.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Añadir número de página a metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update(base_metadata)
                    doc.metadata['page'] = i + 1
                
                all_docs.extend(docs)
                logger.info(f"✓ PDF cargado: {file_path} ({len(docs)} páginas)")
                
            elif lower.endswith(('.md', '.markdown', '.txt')):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata.update(base_metadata)
                
                all_docs.extend(docs)
                logger.info(f"✓ Texto cargado: {file_path}")
                
        except Exception as e:
            logger.error(f"✗ Error cargando {file_path}: {e}")
    
    logger.info(f"Total documentos cargados: {len(all_docs)}")
    return all_docs


def needs_rebuild(docs_dir: str, db_dir: str) -> Tuple[bool, str]:
    """Verifica si se necesita reconstruir el índice"""
    
    manifest_path = os.path.join(db_dir, 'index_manifest.json')
    
    if not os.path.exists(db_dir) or not os.path.exists(manifest_path):
        return True, "Base de datos no existe"
    
    # Cargar manifiesto
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo manifiesto: {e}")
        return True, "Error leyendo manifiesto"
    
    # Archivos actuales
    pattern = os.path.join(docs_dir, "**", "*.*")
    current_files = [
        os.path.normpath(p) for p in glob.glob(pattern, recursive=True)
        if os.path.isfile(p) and p.lower().endswith(('.pdf', '.md', '.markdown', '.txt'))
    ]
    
    # Metadata actual
    current_meta = {p: compute_file_metadata(p) for p in current_files}
    manifest_meta = {
        os.path.normpath(m['path']): m
        for m in manifest.get('files', [])
    }
    
    # Detectar cambios
    current_set = set(current_meta.keys())
    manifest_set = set(manifest_meta.keys())
    
    added = current_set - manifest_set
    removed = manifest_set - current_set
    
    if added:
        return True, f"Archivos añadidos: {len(added)}"
    if removed:
        return True, f"Archivos eliminados: {len(removed)}"
    
    # Detectar modificaciones
    for p in current_set:
        cm = current_meta.get(p)
        mm = manifest_meta.get(p)
        if not mm or cm.get('sha1') != mm.get('sha1'):
            return True, f"Archivo modificado: {p}"
    
    return False, "Índice actualizado"


def save_manifest(docs_dir: str, db_dir: str):
    """Guarda manifiesto de archivos indexados"""
    try:
        pattern = os.path.join(docs_dir, "**", "*.*")
        files = [
            os.path.normpath(p) for p in glob.glob(pattern, recursive=True)
            if os.path.isfile(p) and p.lower().endswith(('.pdf', '.md', '.markdown', '.txt'))
        ]
        
        files_meta = [compute_file_metadata(p) for p in files]
        
        os.makedirs(db_dir, exist_ok=True)
        manifest_path = os.path.join(db_dir, 'index_manifest.json')
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                'files': files_meta,
                'created_at': datetime.now().isoformat(),
                'num_files': len(files_meta)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Manifiesto guardado: {len(files_meta)} archivos")
        
    except Exception as e:
        logger.error(f"Error guardando manifiesto: {e}")


def get_vectorstore():
    """Obtiene o crea el vectorstore con mejoras"""
    
    # Verificar si necesita reconstrucción
    rebuild, reason = needs_rebuild(DOCS_DIR, DB_DIR)
    
    if not rebuild and os.path.exists(DB_DIR):
        logger.info(f"Cargando base de datos existente: {reason}")
        try:
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
            )
            return vectorstore
        except Exception as e:
            logger.error(f"Error cargando base de datos: {e}")
            rebuild = True
            reason = f"Error al cargar: {e}"
    
    # Reconstruir base de datos
    logger.info(f"Reconstruyendo base de datos: {reason}")
    
    # Limpiar directorio anterior
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            logger.info("Base de datos anterior eliminada")
        except Exception as e:
            logger.error(f"Error eliminando base de datos anterior: {e}")
    
    # Cargar documentos
    docs = load_documents_with_metadata(DOCS_DIR)
    
    if not docs:
        logger.warning("No se encontraron documentos para indexar")
        return None
    
    # Chunking semántico
    chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(docs)
    
    logger.info(f"Creando vectorstore con {len(chunks)} chunks...")
    
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=DB_DIR,
        )
        
        # Guardar manifiesto
        save_manifest(DOCS_DIR, DB_DIR)
        
        logger.info(f"✓ Base de datos creada exitosamente")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creando vectorstore: {e}")
        raise


def build_enhanced_prompt(query: str, context: str, sources: List[str]) -> str:
    """Construye un prompt mejorado con instrucciones claras"""
    
    sources_text = "\n".join([f"- {s}" for s in sources])
    
    prompt = f"""Eres un asistente experto en VMware ESXi. Tu objetivo es proporcionar respuestas precisas, técnicas y bien fundamentadas.

CONTEXTO RECUPERADO:
{context}

FUENTES CONSULTADAS:
{sources_text}

PREGUNTA DEL USUARIO:
{query}

INSTRUCCIONES:
1. Responde de forma concisa pero completa en español
2. Basa tu respuesta ÚNICAMENTE en el contexto proporcionado
3. Si el contexto no contiene información suficiente, admítelo claramente
4. Cita las fuentes cuando sea relevante (ej: "Según [nombre_archivo]...")
5. Si hay información conflictiva, menciónalo
6. Usa ejemplos técnicos cuando sea apropiado
7. Mantén un tono profesional y técnico

RESPUESTA:"""
    
    return prompt


def log_retrieval_metrics(metrics: RetrievalMetrics):
    """Registra métricas de retrieval para análisis"""
    try:
        metrics_file = os.path.join(LOGS_DIR, 'retrieval_metrics.jsonl')
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'query': metrics.query,
                'num_chunks': metrics.num_chunks_retrieved,
                'avg_similarity': metrics.avg_similarity_score,
                'sources': metrics.sources_used,
                'context_length': metrics.context_length,
                'method': metrics.retrieval_method
            }, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Error guardando métricas: {e}")


def main():
    """Función principal con todas las mejoras implementadas"""
    
    logger.info("=" * 60)
    logger.info("SISTEMA RAG MEJORADO - VMware ESXi")
    logger.info("=" * 60)
    
    # Obtener vectorstore
    vectorstore = get_vectorstore()
    
    if not vectorstore:
        logger.error("No se pudo crear/cargar la base de datos")
        return
    
    # Cargar documentos para BM25
    logger.info("Preparando retriever híbrido...")
    all_docs = load_documents_with_metadata(DOCS_DIR)
    chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(all_docs)
    
    # Crear retriever híbrido
    hybrid_retriever = HybridRetriever(vectorstore, chunks, alpha=0.6)
    
    # Configurar LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
    
    # Checker de relevancia
    relevance_checker = RelevanceChecker(llm)
    
    logger.info("Sistema listo para consultas")
    print("\n" + "=" * 60)
    print("EXPERTO EN VMWARE ESXi (RAG Mejorado)")
    print("=" * 60)
    print("Comandos especiales:")
    print("  - 'salir' / 'exit' / 'quit': Terminar")
    print("  - 'stats': Ver estadísticas del sistema")
    print("=" * 60 + "\n")
    
    query_count = 0
    
    while True:
        try:
            query = input("\n🔍 Pregunta: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['salir', 'exit', 'quit']:
                logger.info("Sesión terminada por el usuario")
                break
            
            if query.lower() == 'stats':
                # Mostrar estadísticas
                try:
                    manifest_path = os.path.join(DB_DIR, 'index_manifest.json')
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    print(f"\n📊 Estadísticas:")
                    print(f"  - Archivos indexados: {manifest.get('num_files', 0)}")
                    print(f"  - Última actualización: {manifest.get('created_at', 'N/A')}")
                    print(f"  - Consultas realizadas: {query_count}")
                except Exception as e:
                    print(f"Error mostrando estadísticas: {e}")
                continue
            
            query_count += 1
            logger.info(f"Procesando consulta #{query_count}: {query[:100]}...")
            
            # Retrieval híbrido
            print("⏳ Buscando información relevante...")
            results = hybrid_retriever.retrieve(query, k=5)
            
            if not results:
                print("\n⚠️  No se encontró información relevante en la base de datos.")
                continue
            
            # Extraer documentos y calcular métricas
            docs = [doc for doc, score in results]
            scores = [score for doc, score in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Construir contexto
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get('source', 'desconocido')
                page = doc.metadata.get('page', '')
                page_info = f" (página {page})" if page else ""
                
                source_info = f"{os.path.basename(source)}{page_info}"
                if source_info not in sources:
                    sources.append(source_info)
                
                context_parts.append(
                    f"--- Fragmento {i} [{source_info}] (relevancia: {score:.2f}) ---\n"
                    f"{doc.page_content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Verificar relevancia
            is_relevant, relevance_msg = relevance_checker.check_relevance(query, context)
            
            if not is_relevant:
                logger.warning(f"Contexto no relevante: {relevance_msg}")
                print(f"\n⚠️  {relevance_msg}")
                print("Intenta reformular tu pregunta.")
                continue
            
            # Log de métricas
            metrics = RetrievalMetrics(
                query=query,
                num_chunks_retrieved=len(docs),
                avg_similarity_score=avg_score,
                sources_used=sources,
                context_length=len(context),
                retrieval_method="hybrid"
            )
            log_retrieval_metrics(metrics)
            
            # Construir prompt mejorado
            prompt = build_enhanced_prompt(query, context, sources)
            
            # Generar respuesta
            print("💭 Generando respuesta...\n")
            
            try:
                ai_message = llm.invoke(prompt)
                response = getattr(ai_message, 'content', str(ai_message))
                
                # Mostrar respuesta
                print("─" * 60)
                print("📄 RESPUESTA:")
                print("─" * 60)
                print(response)
                print("─" * 60)
                
                # Mostrar fuentes
                print(f"\n📚 Fuentes consultadas ({len(sources)}):")
                for source in sources:
                    print(f"  • {source}")
                
                # Mostrar métricas de calidad
                print(f"\n📈 Calidad del retrieval:")
                print(f"  • Chunks recuperados: {len(docs)}")
                print(f"  • Relevancia promedio: {avg_score:.2%}")
                print(f"  • Longitud del contexto: {len(context):,} caracteres")
                
                logger.info(f"Consulta #{query_count} completada exitosamente")
                
            except Exception as e:
                logger.error(f"Error generando respuesta: {e}")
                print(f"\n❌ Error al generar respuesta: {e}")
        
        except KeyboardInterrupt:
            print("\n\nSesión interrumpida por el usuario")
            logger.info("Sesión interrumpida (Ctrl+C)")
            break
        
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
            print(f"\n❌ Error inesperado: {e}")
    
    print("\n👋 ¡Hasta luego!")
    logger.info(f"Sesión finalizada. Total de consultas: {query_count}")


if __name__ == "__main__":
    main()
