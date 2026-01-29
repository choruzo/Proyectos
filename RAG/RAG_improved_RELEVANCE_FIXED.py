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

# Fix para Windows: forzar UTF-8 en stdout/stderr
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'rag_{datetime.now().strftime("%Y%m%d")}.log'), encoding='utf-8'),
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
    reranked: bool = False
    query_expanded: bool = False


# ============================================================================
# MEJORA 1: QUERY EXPANSION - Enriquecer queries cortas
# ============================================================================
class QueryExpander:
    """Expande queries cortas para mejorar retrieval"""
    
    def __init__(self, llm):
        self.llm = llm
        self.cache = {}  # Cache de expansiones
    
    def should_expand(self, query: str) -> bool:
        """Determina si una query necesita expansión"""
        words = query.split()
        
        # Expandir si:
        # - Muy corta (< 5 palabras)
        # - Muy genérica (palabras comunes)
        if len(words) < 5:
            return True
        
        # Verificar si tiene términos técnicos específicos
        technical_terms = {'esxi', 'vcenter', 'vmware', 'vsphere', 'datastore', 'vmotion'}
        has_technical = any(term in query.lower() for term in technical_terms)
        
        return not has_technical
    
    def expand(self, query: str) -> str:
        """Expande la query con términos técnicos y sinónimos"""
        
        # Verificar cache
        if query in self.cache:
            logger.info(f"Query expansion (cache): {query} -> {self.cache[query]}")
            return self.cache[query]
        
        if not self.should_expand(query):
            return query
        
        logger.info(f"Expandiendo query: {query}")
        
        prompt = f"""Eres un experto en VMware ESXi. Reformula esta pregunta añadiendo términos técnicos relevantes y sinónimos.

Pregunta original: {query}

Reglas:
1. Mantén el significado original
2. Añade términos técnicos VMware relacionados
3. Incluye sinónimos (ej: "VM" = "máquina virtual" = "virtual machine")
4. Máximo 15 palabras adicionales
5. Responde SOLO con la pregunta expandida, sin explicaciones

Pregunta expandida:"""

        try:
            response = self.llm.invoke(prompt)
            expanded = getattr(response, 'content', str(response)).strip()
            
            # Validar que la expansión tenga sentido
            if len(expanded) > 0 and len(expanded) < len(query) * 3:
                self.cache[query] = expanded
                logger.info(f"Query expandida: {expanded}")
                return expanded
            else:
                logger.warning("Expansión inválida, usando query original")
                return query
                
        except Exception as e:
            logger.error(f"Error expandiendo query: {e}")
            return query


# ============================================================================
# MEJORA 2: RERANKING - Reordenar resultados por relevancia real
# ============================================================================
class CrossEncoderReranker:
    """Reranker usando el LLM para evaluar relevancia"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def rerank(self, query: str, docs: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Reordena documentos por relevancia usando LLM"""
        
        if len(docs) <= top_k:
            return docs
        
        logger.info(f"Reranking {len(docs)} documentos...")
        
        # Evaluar cada documento
        scored_docs = []
        
        for doc, original_score in docs[:15]:  # Solo rerank top 15 por eficiencia
            relevance_score = self._score_relevance(query, doc)
            
            # Combinar score original con relevance score
            final_score = (original_score * 0.3) + (relevance_score * 0.7)
            scored_docs.append((doc, final_score))
        
        # Ordenar por nuevo score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranking completado: top score = {scored_docs[0][1]:.3f}")
        return scored_docs[:top_k]
    
    def _score_relevance(self, query: str, doc: Document) -> float:
        """Evalúa relevancia de un documento para una query"""
        
        # Truncar documento si es muy largo
        content = doc.page_content[:800]
        
        prompt = f"""Evalúa la relevancia de este fragmento para responder la pregunta.

PREGUNTA: {query}

FRAGMENTO:
{content}

¿Qué tan relevante es este fragmento para responder la pregunta?
Responde SOLO con un número del 0 al 10:
- 0 = completamente irrelevante
- 5 = parcialmente relevante
- 10 = extremadamente relevante, responde directamente

Puntuación:"""

        try:
            response = self.llm.invoke(prompt)
            score_text = getattr(response, 'content', str(response)).strip()
            
            # Extraer número
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                return min(max(score / 10.0, 0.0), 1.0)  # Normalizar 0-1
            else:
                return 0.5  # Default si no puede parsear
                
        except Exception as e:
            logger.error(f"Error scoring relevance: {e}")
            return 0.5


# ============================================================================
# MEJORA 3: CHUNKING ADAPTATIVO - Tamaño óptimo basado en análisis
# ============================================================================
class AdaptiveSemanticChunker:
    """Chunking mejorado con tamaños adaptativos"""
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 250):
        """
        Tamaños aumentados para mejor contexto:
        - chunk_size: 1000 -> 1200 (20% más contexto)
        - chunk_overlap: 200 -> 250 (25% más overlap)
        """
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
        
        logger.info(f"Documentos divididos en {len(chunks)} chunks semánticos (adaptativo)")
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


# ============================================================================
# BM25 Retriever (sin cambios, funciona bien)
# ============================================================================
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
        logger.info(f"Construyendo índice BM25 para {self.N} documentos...")
        
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
        
        logger.info(f"[OK] Índice BM25 completado ({len(self.idf)} términos únicos)")
    
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


# ============================================================================
# MEJORA 4: HYBRID RETRIEVER CON ALPHA ADAPTATIVO
# ============================================================================
class ImprovedHybridRetriever:
    """Retriever híbrido mejorado con alpha adaptativo"""
    
    def __init__(self, vectorstore, documents: List[Document], base_alpha: float = 0.6):
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever(documents)
        self.base_alpha = base_alpha
    
    def _calculate_adaptive_alpha(self, query: str) -> float:
        """Calcula alpha dinámicamente basado en características de la query"""
        
        words = query.split()
        num_words = len(words)
        
        # Queries cortas (< 5 palabras) -> más peso a BM25 (keywords)
        # Queries largas (> 10 palabras) -> más peso a vectorial (semántica)
        if num_words < 5:
            alpha = 0.4  # 40% vectorial, 60% BM25
        elif num_words > 10:
            alpha = 0.75  # 75% vectorial, 25% BM25
        else:
            alpha = self.base_alpha
        
        logger.info(f"Alpha adaptativo: {alpha:.2f} (query: {num_words} palabras)")
        return alpha
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieval híbrido con normalización de scores y alpha adaptativo"""
        
        # Calcular alpha dinámicamente
        alpha = self._calculate_adaptive_alpha(query)
        
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
                'vector_score': score * alpha,
                'bm25_score': 0.0
            }
        
        for doc, score in bm25_normalized:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['bm25_score'] = score * (1 - alpha)
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'bm25_score': score * (1 - alpha)
                }
        
        # Calcular score final y ordenar
        final_results = []
        for doc_id, data in combined_scores.items():
            final_score = data['vector_score'] + data['bm25_score']
            final_results.append((data['doc'], final_score))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Retrieval híbrido: {len(final_results)} resultados combinados")
        return final_results[:k]


# ============================================================================
# MEJORA 5: RELEVANCE CHECKER MÁS ESTRICTO
# ============================================================================
class StrictRelevanceChecker:
    """Verifica si el contexto recuperado es relevante para la pregunta"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def check_relevance(self, query: str, context: str, min_score: float = 0.4) -> Tuple[bool, str, float]:
        """Verifica relevancia del contexto con score numérico"""
        
        # Verificación rápida por keywords
        query_tokens = set(query.lower().split())
        context_tokens = set(context.lower().split())
        overlap = len(query_tokens & context_tokens) / len(query_tokens) if query_tokens else 0
        
        logger.info(f"Keyword overlap: {overlap:.2%}")
        
        if overlap < 0.05:  # Muy poco overlap
            return False, "El contexto no está relacionado con la pregunta (sin keywords comunes)", overlap
        
        # Si hay buen overlap, verificar con LLM para mayor precisión
        if overlap > 0.15:
            llm_score = self._llm_check_relevance(query, context[:1500])  # Limitar contexto
            
            if llm_score >= min_score:
                return True, f"Contexto relevante (overlap: {overlap:.1%}, LLM: {llm_score:.1%})", llm_score
            else:
                return False, f"Contexto no suficientemente relevante (LLM score: {llm_score:.1%})", llm_score
        
        # Overlap bajo pero no cero - requiere verificación LLM
        llm_score = self._llm_check_relevance(query, context[:1500])
        
        if llm_score >= min_score:
            return True, f"Contexto aceptable (LLM: {llm_score:.1%})", llm_score
        else:
            return False, f"Contexto insuficiente (overlap: {overlap:.1%}, LLM: {llm_score:.1%})", llm_score
    
    def _llm_check_relevance(self, query: str, context: str) -> float:
        """Usa LLM para verificar relevancia"""
        
        prompt = f"""Evalúa si este contexto puede responder la pregunta.

PREGUNTA: {query}

CONTEXTO:
{context}

¿El contexto contiene información relevante para responder la pregunta?
Responde SOLO con un número del 0 al 10:
- 0-3 = irrelevante
- 4-6 = parcialmente relevante
- 7-10 = muy relevante

Puntuación:"""

        try:
            response = self.llm.invoke(prompt)
            score_text = getattr(response, 'content', str(response)).strip()
            
            # Extraer número
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                return min(max(score / 10.0, 0.0), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error en LLM relevance check: {e}")
            return 0.5


# ============================================================================
# Funciones auxiliares (sin cambios significativos)
# ============================================================================

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
        logger.error(f"Directorio {docs_dir} no existe")
        return all_docs
    
    # Buscar archivos soportados
    supported_extensions = ['*.pdf', '*.md', '*.markdown', '*.txt']
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(docs_dir, '**', ext), recursive=True))
    
    logger.info(f"Encontrados {len(files)} archivos para procesar")
    
    for file_path in files:
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                logger.info(f"[OK] PDF cargado: {file_path} ({len(docs)} páginas)")
            
            elif ext in ['.md', '.markdown', '.txt']:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                logger.info(f"[OK] Texto cargado: {file_path}")
            
            else:
                continue
            
            # Enriquecer metadata
            for doc in docs:
                doc.metadata['file_type'] = ext
                doc.metadata['file_name'] = os.path.basename(file_path)
            
            all_docs.extend(docs)
            
        except Exception as e:
            logger.error(f"Error cargando {file_path}: {e}")
    
    logger.info(f"Total documentos cargados: {len(all_docs)}")
    return all_docs


class VectorStoreManager:
    """Gestiona el ciclo de vida del vectorstore con detección de cambios"""
    
    def __init__(self, db_dir: str, docs_dir: str, embedding_model: str):
        self.db_dir = db_dir
        self.docs_dir = docs_dir
        self.embedding_model = embedding_model
        self.manifest_path = os.path.join(db_dir, 'index_manifest.json')
    
    def needs_rebuild(self) -> Tuple[bool, str]:
        """Verifica si la base de datos necesita reconstrucción"""
        
        if not os.path.exists(self.db_dir):
            return True, "Base de datos no existe"
        
        if not os.path.exists(self.manifest_path):
            return True, "Manifiesto no existe"
        
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            stored_files = {f['path']: f for f in manifest.get('files', [])}
            
            # Obtener archivos actuales
            current_files = {}
            for ext in ['*.pdf', '*.md', '*.markdown', '*.txt']:
                for path in glob.glob(os.path.join(self.docs_dir, '**', ext), recursive=True):
                    current_files[os.path.normpath(path)] = compute_file_metadata(path)
            
            # Comparar
            if set(stored_files.keys()) != set(current_files.keys()):
                return True, f"Archivos cambiaron: {len(stored_files)} -> {len(current_files)}"
            
            # Verificar hashes
            for path, current_meta in current_files.items():
                stored_meta = stored_files.get(path, {})
                if current_meta.get('sha1') != stored_meta.get('sha1'):
                    return True, f"Archivo modificado: {path}"
            
            return False, "Base de datos actualizada"
            
        except Exception as e:
            logger.error(f"Error verificando manifest: {e}")
            return True, f"Error en verificación: {e}"
    
    def save_manifest(self, files_metadata: List[Dict]):
        """Guarda manifiesto de archivos indexados"""
        try:
            manifest = {
                'created_at': datetime.now().isoformat(),
                'num_files': len(files_metadata),
                'files': files_metadata
            }
            
            os.makedirs(self.db_dir, exist_ok=True)
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Manifiesto guardado: {len(files_metadata)} archivos")
            
        except Exception as e:
            logger.error(f"Error guardando manifiesto: {e}")


def get_vectorstore() -> Optional[Chroma]:
    """Obtiene o crea vectorstore con gestión inteligente de actualizaciones"""
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    manager = VectorStoreManager(DB_DIR, DOCS_DIR, EMBEDDING_MODEL)
    
    needs_rebuild, reason = manager.needs_rebuild()
    
    if needs_rebuild:
        logger.info(f"Reconstruyendo base de datos: {reason}")
        
        # Eliminar DB anterior
        if os.path.exists(DB_DIR):
            try:
                deleted_files = len([f for f in Path(DB_DIR).rglob('*') if f.is_file()])
                shutil.rmtree(DB_DIR)
                logger.info(f"Base de datos anterior eliminada ({deleted_files} archivos)")
            except Exception as e:
                logger.error(f"Error eliminando DB: {e}")
        
        # Cargar documentos
        all_docs = load_documents_with_metadata(DOCS_DIR)
        
        if not all_docs:
            logger.error("No hay documentos para indexar")
            return None
        
        # Chunking mejorado
        logger.info("Dividiendo documentos en chunks semánticos...")
        chunker = AdaptiveSemanticChunker(chunk_size=1200, chunk_overlap=250)
        chunks = chunker.split_documents(all_docs)
        
        logger.info(f"Documentos divididos en {len(chunks)} chunks")
        
        # Crear vectorstore
        return create_vectorstore(chunks, embeddings, manager)
    
    else:
        logger.info(f"Cargando base de datos existente: {reason}")
        try:
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
            logger.info("[OK] Base de datos cargada")
            return vectorstore
        except Exception as e:
            logger.error(f"Error cargando DB: {e}")
            return None


def create_vectorstore(chunks: List[Document], embeddings, manager: VectorStoreManager) -> Chroma:
    """Crea vectorstore con progreso visible"""
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Creando vectorstore con {len(chunks)} chunks...")
        logger.info("NOTA: La creación de embeddings puede tardar varios minutos...")
        logger.info(f"      Procesando aproximadamente {len(chunks)} fragmentos...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[OK] Base de datos creada en {elapsed:.1f} segundos")
        
        # Guardar manifiesto
        files_metadata = []
        for ext in ['*.pdf', '*.md', '*.markdown', '*.txt']:
            for path in glob.glob(os.path.join(DOCS_DIR, '**', ext), recursive=True):
                files_metadata.append(compute_file_metadata(path))
        
        manager.save_manifest(files_metadata)
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creando vectorstore: {e}")
        raise


def build_enhanced_prompt(query: str, context: str, sources: List[str]) -> str:
    """Construye un prompt más estricto que evita alucinaciones"""
    
    sources_text = "\n".join([f"- {s}" for s in sources])
    
    prompt = f"""Eres un asistente experto en VMware ESXi. Tu objetivo es proporcionar respuestas precisas basadas ÚNICAMENTE en el contexto proporcionado.

CONTEXTO RECUPERADO:
{context}

FUENTES CONSULTADAS:
{sources_text}

PREGUNTA DEL USUARIO:
{query}

INSTRUCCIONES CRÍTICAS:
1. Lee el contexto cuidadosamente
2. Si el contexto CONTIENE la respuesta → responde de forma concisa y técnica
3. Si el contexto NO CONTIENE la respuesta → di EXACTAMENTE: "No encontré esta información en la documentación proporcionada."
4. NUNCA inventes información o des respuestas genéricas no basadas en el contexto
5. Cita fragmentos textuales cuando sea posible (ej: "según el documento...")
6. Usa español técnico y profesional

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
                'method': metrics.retrieval_method,
                'reranked': metrics.reranked,
                'query_expanded': metrics.query_expanded
            }, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Error guardando métricas: {e}")


# ============================================================================
# MAIN - Integración de todas las mejoras
# ============================================================================

def main():
    """Función principal con todas las mejoras de relevancia integradas"""
    
    logger.info("=" * 60)
    logger.info("SISTEMA RAG MEJORADO v2.0 - VMware ESXi")
    logger.info("MEJORAS: Query Expansion + Reranking + Alpha Adaptativo")
    logger.info("=" * 60)
    
    # Configurar LLM (necesario para query expansion y reranking)
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
    
    # Obtener vectorstore
    vectorstore = get_vectorstore()
    
    if not vectorstore:
        logger.error("No se pudo crear/cargar la base de datos")
        return
    
    # Cargar documentos para BM25
    logger.info("Preparando retriever híbrido mejorado...")
    all_docs = load_documents_with_metadata(DOCS_DIR)
    chunker = AdaptiveSemanticChunker(chunk_size=1200, chunk_overlap=250)
    chunks = chunker.split_documents(all_docs)
    
    # Crear componentes mejorados
    hybrid_retriever = ImprovedHybridRetriever(vectorstore, chunks, base_alpha=0.6)
    query_expander = QueryExpander(llm)
    reranker = CrossEncoderReranker(llm)
    relevance_checker = StrictRelevanceChecker(llm)
    
    logger.info("✓ Sistema listo con mejoras de relevancia activadas")
    print("\n" + "=" * 70)
    print("EXPERTO EN VMWARE ESXi (RAG Mejorado v2.0)")
    print("=" * 70)
    print("Mejoras activadas:")
    print("  ✓ Query Expansion (queries cortas se expanden automáticamente)")
    print("  ✓ Reranking con LLM (resultados reordenados por relevancia)")
    print("  ✓ Alpha Adaptativo (pesos dinámicos según tipo de query)")
    print("  ✓ Chunking optimizado (1200 chars, overlap 250)")
    print("\nComandos especiales:")
    print("  - 'salir' / 'exit' / 'quit': Terminar")
    print("  - 'stats': Ver estadísticas del sistema")
    print("=" * 70 + "\n")
    
    query_count = 0
    
    while True:
        try:
            query = input("\n[?] Pregunta: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['salir', 'exit', 'quit']:
                logger.info("Sesión terminada por el usuario")
                break
            
            if query.lower() == 'stats':
                try:
                    manifest_path = os.path.join(DB_DIR, 'index_manifest.json')
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    print(f"\n[STATS] Estadísticas:")
                    print(f"  - Archivos indexados: {manifest.get('num_files', 0)}")
                    print(f"  - Última actualización: {manifest.get('created_at', 'N/A')}")
                    print(f"  - Consultas realizadas: {query_count}")
                    print(f"  - Mejoras activas: Query Expansion, Reranking, Alpha Adaptativo")
                except Exception as e:
                    print(f"Error mostrando estadísticas: {e}")
                continue
            
            query_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"CONSULTA #{query_count}: {query}")
            logger.info(f"{'='*60}")
            
            # MEJORA 1: Query Expansion
            query_expanded = query_expander.expand(query)
            was_expanded = query_expanded != query
            
            if was_expanded:
                print(f"[*] Query expandida: {query_expanded}")
            
            # Retrieval híbrido (con alpha adaptativo automático)
            print("[...] Buscando información relevante...")
            results = hybrid_retriever.retrieve(query_expanded, k=15)  # Más resultados para reranking
            
            if not results:
                print("\n[!] No se encontró información relevante en la base de datos.")
                continue
            
            # MEJORA 2: Reranking
            print("[...] Reordenando resultados por relevancia...")
            results = reranker.rerank(query, results, top_k=5)
            
            # Extraer documentos y calcular métricas
            docs = [doc for doc, score in results]
            scores = [score for doc, score in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            logger.info(f"Relevancia promedio después de reranking: {avg_score:.2%}")
            
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
            
            # MEJORA 3: Verificación de relevancia más estricta
            is_relevant, relevance_msg, relevance_score = relevance_checker.check_relevance(query, context, min_score=0.4)
            
            if not is_relevant:
                logger.warning(f"Contexto rechazado: {relevance_msg}")
                print(f"\n[!] {relevance_msg}")
                print("Intenta reformular tu pregunta con más detalles técnicos.")
                continue
            
            logger.info(f"Contexto aceptado: {relevance_msg}")
            
            # Log de métricas mejoradas
            metrics = RetrievalMetrics(
                query=query,
                num_chunks_retrieved=len(docs),
                avg_similarity_score=avg_score,
                sources_used=sources,
                context_length=len(context),
                retrieval_method="hybrid+reranking",
                reranked=True,
                query_expanded=was_expanded
            )
            log_retrieval_metrics(metrics)
            
            # Construir prompt mejorado (más estricto contra alucinaciones)
            prompt = build_enhanced_prompt(query, context, sources)
            
            # Generar respuesta
            print("[AI] Generando respuesta...\n")
            
            try:
                ai_message = llm.invoke(prompt)
                response = getattr(ai_message, 'content', str(ai_message))
                
                # Mostrar respuesta
                print("=" * 70)
                print("RESPUESTA:")
                print("=" * 70)
                print(response)
                print("=" * 70)
                
                # Mostrar fuentes
                print(f"\n[FUENTES] Consultadas ({len(sources)}):")
                for source in sources:
                    print(f"  * {source}")
                
                # Mostrar métricas mejoradas
                print(f"\n[METRICAS] Calidad del retrieval:")
                print(f"  * Chunks recuperados: {len(docs)}")
                print(f"  * Relevancia promedio: {avg_score:.2%} ⬆")
                print(f"  * Score de relevancia: {relevance_score:.2%}")
                print(f"  * Longitud del contexto: {len(context):,} caracteres")
                print(f"  * Query expandida: {'Sí' if was_expanded else 'No'}")
                print(f"  * Reranking aplicado: Sí")
                
                logger.info(f"Consulta #{query_count} completada exitosamente")
                
            except Exception as e:
                logger.error(f"Error generando respuesta: {e}")
                print(f"\n[ERROR] Error al generar respuesta: {e}")
        
        except KeyboardInterrupt:
            print("\n\nSesión interrumpida por el usuario")
            logger.info("Sesión interrumpida (Ctrl+C)")
            break
        
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
            print(f"\n[ERROR] Error inesperado: {e}")
    
    print("\n=== Hasta luego! ===")
    logger.info(f"Sesión finalizada. Total de consultas: {query_count}")


if __name__ == "__main__":
    main()
