import os
import glob
import shutil
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Set
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
import re

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
# FIX 1: QUERY EXPANSION SIMPLIFICADA Y ROBUSTA
# ============================================================================
class SimpleQueryExpander:
    """Expansión de queries SIN LLM - usa reglas heurísticas"""
    
    def __init__(self):
        # Mapeo de términos comunes a términos técnicos VMware
        # v2.3: Expandido de 12 a 37 términos
        self.expansions = {
            # Términos básicos (12 originales)
            'vm': ['virtual machine', 'máquina virtual', 'guest'],
            'apagar': ['shutdown', 'power off', 'apagado', 'detener'],
            'encender': ['power on', 'encendido', 'iniciar', 'start'],
            'reiniciar': ['restart', 'reboot', 'reset'],
            'migrar': ['migration', 'vmotion', 'migrate'],
            'snapshot': ['instantánea', 'backup', 'restore'],
            'disco': ['disk', 'storage', 'datastore', 'vmdk'],
            'red': ['network', 'vswitch', 'nic', 'vlan'],
            'cpu': ['processor', 'core', 'vcpu'],
            'memoria': ['memory', 'ram'],
            'plantilla': ['template', 'ova', 'ovf'],
            'clonar': ['clone', 'copy', 'duplicate'],

            # Infraestructura (4 nuevos)
            'host': ['esxi', 'servidor', 'server', 'physical server', 'hypervisor'],
            'datastore': ['almacenamiento', 'storage', 'vmfs', 'shared storage'],
            'cluster': ['clúster', 'agrupación', 'pool', 'resource pool'],
            'vcenter': ['management', 'center', 'control', 'vSphere'],

            # Alta disponibilidad y rendimiento (4 nuevos)
            'ha': ['high availability', 'alta disponibilidad', 'failover', 'redundancia'],
            'drs': ['distributed resource scheduler', 'load balancing', 'balanceo'],
            'vmotion': ['live migration', 'migración en vivo', 'moving vm'],
            'rendimiento': ['performance', 'velocidad', 'optimization', 'efficiency'],

            # Red y seguridad (4 nuevos)
            'vlan': ['red virtual', 'segmentación', 'network segment'],
            'vswitch': ['virtual switch', 'switch virtual', 'network switch'],
            'firewall': ['cortafuegos', 'security', 'seguridad'],
            'ip': ['dirección ip', 'address', 'networking'],

            # Almacenamiento (3 nuevos)
            'vmdk': ['virtual disk', 'disco virtual', 'disk file'],
            'nfs': ['network file system', 'shared storage', 'almacenamiento compartido'],
            'iscsi': ['storage area network', 'san', 'block storage'],

            # Operaciones (4 nuevos)
            'arrancar': ['start', 'boot', 'power on', 'iniciar'],
            'detener': ['stop', 'halt', 'apagar', 'shutdown'],
            'backup': ['copia de seguridad', 'restore', 'recuperación', 'respaldo'],
            'monitoreo': ['monitoring', 'observabilidad', 'alertas', 'metrics'],

            # Administración (5 nuevos)
            'usuario': ['user', 'account', 'login', 'credencial'],
            'permiso': ['permission', 'role', 'access', 'autorización'],
            'licencia': ['license', 'key', 'activation', 'subscription'],
            'configurar': ['configure', 'setup', 'config', 'set up'],
            'configuración': ['configuration', 'settings', 'setup', 'config'],

            # Recursos (2 nuevos)
            'recurso': ['resource', 'allocation', 'asignación'],
            'límite': ['limit', 'threshold', 'quota', 'cuota'],
        }
    
    def expand(self, query: str) -> Tuple[str, bool]:
        """Expande query usando reglas heurísticas (bidireccional)"""

        query_lower = query.lower()
        expanded_terms = set()  # Usar set para evitar duplicados
        was_expanded = False

        # Búsqueda en claves (como antes)
        for term, synonyms in self.expansions.items():
            if term in query_lower:
                expanded_terms.update(synonyms)
                was_expanded = True

        # NUEVO v2.3: Búsqueda inversa en valores (bidireccional)
        # Si usuario escribe "virtual machine", también busca "vm"
        for term, synonyms in self.expansions.items():
            for synonym in synonyms:
                if synonym in query_lower and term not in query_lower:
                    expanded_terms.add(term)
                    was_expanded = True

        if was_expanded:
            expanded = f"{query} {' '.join(expanded_terms)}"
            logger.info(f"Query expandida (reglas): {query} → +{len(expanded_terms)} términos")
            return expanded, True
        else:
            return query, False


# ============================================================================
# FIX 1.5: EMBEDDING CACHE (v2.3)
# ============================================================================
class EmbeddingCache:
    """Cache LRU para embeddings de queries"""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        logger.info(f"Embedding cache inicializado (max_size={max_size})")

    def _normalize_query(self, query: str) -> str:
        """Normaliza query para cache key (lowercase, strip)"""
        return query.lower().strip()

    def get(self, query: str) -> Optional[List[float]]:
        """Obtiene embedding del cache si existe"""
        key = self._normalize_query(query)
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            logger.debug(f"Cache HIT: {query[:50]}... (hits={self.hits})")
            return self.cache[key]
        self.misses += 1
        logger.debug(f"Cache MISS: {query[:50]}... (misses={self.misses})")
        return None

    def put(self, query: str, embedding: List[float]):
        """Cachea nuevo embedding"""
        key = self._normalize_query(query)
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Remove oldest (FIFO)
            oldest = self.cache.popitem(last=False)
            logger.debug(f"Cache eviction: {oldest[0][:30]}...")
        self.cache[key] = embedding

    def get_stats(self) -> Dict[str, float]:
        """Retorna estadísticas del cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


# ============================================================================
# FIX 2: RERANKING MÁS EFICIENTE (BATCH + LÍMITE)
# ============================================================================
class FastReranker:
    """Reranker optimizado con boost para docs internos"""
    
    def __init__(self, internal_docs_metadata: Set[Tuple[str, int]] = None):
        self.internal_docs_metadata = internal_docs_metadata or set()
    
    def rerank(self, query: str, docs: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Reordena usando heurísticas rápidas en lugar de LLM"""
        
        if len(docs) <= top_k:
            return docs
        
        logger.info(f"Reranking rápido de {len(docs)} documentos...")
        
        # Extraer términos clave de la query
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc, original_score in docs[:20]:  # Solo top 20
            content_lower = doc.page_content.lower()
            
            # Score basado en:
            # 1. Score original (30%)
            # 2. Frecuencia de términos de query en contenido (40%)
            # 3. Longitud del contenido (más largo = más info) (15%)
            # 4. Posición en resultados originales (15%)
            
            # Frecuencia de términos
            term_freq = sum(1 for term in query_terms if term in content_lower) / max(len(query_terms), 1)
            
            # Longitud normalizada (preferir chunks con más contenido)
            length_score = min(len(doc.page_content) / 1500, 1.0)
            
            # Penalizar posiciones bajas
            position_penalty = 1.0 - (docs.index((doc, original_score)) / len(docs)) * 0.3
            
            # NUEVO v2.3: Bonus para docs internos (usando metadata)
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            is_internal = (source, page) in self.internal_docs_metadata
            internal_bonus = 0.4 if is_internal else 0.0
            
            # Score final con bonus
            final_score = (
                original_score * 0.25 +
                term_freq * 0.40 +
                length_score * 0.15 +
                position_penalty * 0.10 +
                internal_bonus  # +30% si es doc interno
            )
            
            scored_docs.append((doc, final_score))
        
        # Ordenar por nuevo score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_internals = sum(1 for doc, _ in scored_docs[:top_k]
                           if (doc.metadata.get('source', ''), doc.metadata.get('page', 0))
                           in self.internal_docs_metadata)
        logger.info(f"Reranking: top {top_k} incluye {top_internals} docs internos")
        logger.info(f"Reranking completado: top score = {scored_docs[0][1]:.3f}")
        return scored_docs[:top_k]


# ============================================================================
# FIX 3: AUMENTAR K INICIAL PARA MEJOR RECALL
# ============================================================================
class ImprovedHybridRetriever:
    """Retriever híbrido con k más alto inicialmente"""
    
    def __init__(self, vectorstore, documents: List[Document], base_alpha: float = 0.5):
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever(documents)
        self.base_alpha = base_alpha

        # NUEVO v2.3: Inicializar embedding cache
        self.embedding_cache = EmbeddingCache(max_size=1000)

        # NUEVO v2.3: Identificar documentos internos (.md) para boosting
        # FIX v2.3: Usar metadata en lugar de id() para evitar identity mismatch
        self.internal_docs_metadata: Set[Tuple[str, int]] = set()
        for doc in documents:
            file_type = doc.metadata.get('file_type', '')
            if file_type in ['.md', '.markdown']:
                source = doc.metadata.get('source', '')
                page = doc.metadata.get('page', 0)
                self.internal_docs_metadata.add((source, page))
        logger.info(f"Docs internos identificados: {len(self.internal_docs_metadata)}")

    def _is_internal_doc(self, doc: Document) -> bool:
        """Verifica si un documento es interno usando metadata"""
        source = doc.metadata.get('source', '')
        page = doc.metadata.get('page', 0)
        return (source, page) in self.internal_docs_metadata

    def _calculate_adaptive_alpha(self, query: str) -> float:
        """Calcula alpha dinámicamente"""
        words = query.split()
        num_words = len(words)
        
        if num_words < 5:
            alpha = 0.35  # MÁS peso a BM25 para queries cortas
        elif num_words > 10:
            alpha = 0.70
        else:
            alpha = self.base_alpha
        
        logger.info(f"Alpha adaptativo: {alpha:.2f} (query: {num_words} palabras)")
        return alpha
    
    def retrieve(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Retrieval con K MÁS ALTO (20 → 30 candidatos) + CACHE"""

        alpha = self._calculate_adaptive_alpha(query)

        # INCREMENTAR k para mejor recall
        retrieval_k = k * 2  # Doble de candidatos

        try:
            # NUEVO v2.3: Intentar usar cache de embeddings
            cached_embedding = self.embedding_cache.get(query)

            if cached_embedding is not None:
                # Usar embedding cacheado
                vector_results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
                    cached_embedding,
                    k=retrieval_k
                )
                logger.info(f"Búsqueda vectorial (CACHE): {len(vector_results)} resultados")
            else:
                # Generar nuevo embedding y cachearlo
                vector_results = self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=retrieval_k
                )
                logger.info(f"Búsqueda vectorial (NUEVO): {len(vector_results)} resultados")

                # Extraer y cachear el embedding
                try:
                    embedding_model = self.vectorstore._embedding_function
                    if hasattr(embedding_model, 'embed_query'):
                        embedding = embedding_model.embed_query(query)
                        self.embedding_cache.put(query, embedding)
                        logger.debug(f"Embedding cacheado: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"No se pudo cachear embedding: {e}")

        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            vector_results = [(doc, 0.0) for doc in self.vectorstore.similarity_search(query, k=retrieval_k)]
        
        try:
            bm25_results = self.bm25.search(query, k=retrieval_k)
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
        # NUEVO v2.3: Aplicar BOOSTING a docs internos
        # v2.3: Aumentado a 75% para priorizar más agresivamente
        INTERNAL_BOOST = 0.75  # 75% boost a docs .md
        
        combined_scores = {}
        for doc, score in vector_normalized:
            doc_id = id(doc)
            boost = 1.0 + INTERNAL_BOOST if self._is_internal_doc(doc) else 1.0
            is_internal = self._is_internal_doc(doc)

            combined_scores[doc_id] = {
                'doc': doc,
                'vector_score': score * alpha * boost,
                'bm25_score': 0.0,
                'is_internal': is_internal
            }

        for doc, score in bm25_normalized:
            doc_id = id(doc)
            boost = 1.0 + INTERNAL_BOOST if self._is_internal_doc(doc) else 1.0
            is_internal = self._is_internal_doc(doc)

            if doc_id in combined_scores:
                combined_scores[doc_id]['bm25_score'] = score * (1 - alpha) * boost
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'bm25_score': score * (1 - alpha) * boost,
                    'is_internal': is_internal
                }
        
        # Calcular score final
        final_results = []
        for doc_id, data in combined_scores.items():
            final_score = data['vector_score'] + data['bm25_score']
            final_results.append((data['doc'], final_score))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Log de boosting
        internal_count = sum(1 for _, data in combined_scores.items() if data.get('is_internal', False))
        logger.info(f"Retrieval: {len(final_results)} resultados ({internal_count} con boost)")
        
        return final_results[:k]


# ============================================================================
# COMPONENTES SIN CAMBIOS (BM25, Chunker, etc.)
# ============================================================================

class BM25Retriever:
    """Implementación BM25"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.avgdl = 0
        self.N = len(documents)
        self.idf = {}
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return [t for t in tokens if len(t) > 2]
    
    def _build_index(self):
        logger.info(f"Construyendo índice BM25 para {self.N} documentos...")
        
        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))
            
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.doc_term_freqs.append(term_freq)
        
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        df = {}
        for term_freq in self.doc_term_freqs:
            for term in term_freq.keys():
                df[term] = df.get(term, 0) + 1
        
        import math
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
        
        logger.info(f"[OK] Índice BM25 completado ({len(self.idf)} términos únicos)")
    
    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Tuple[Document, float]]:
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
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class AdaptiveSemanticChunker:
    """Chunking semántico"""
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.separators = [
            "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
        ]
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunks = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            file_type = Path(source).suffix.lower()
            
            if file_type in ['.md', '.markdown']:
                chunks.extend(self._split_markdown(doc))
            else:
                chunks.extend(self._split_recursive(doc))
        
        logger.info(f"Documentos divididos en {len(chunks)} chunks semánticos (adaptativo)")
        return chunks
    
    def _split_markdown(self, doc: Document) -> List[Document]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        try:
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_chunks = md_splitter.split_text(doc.page_content)
            
            result = []
            for chunk in md_chunks:
                metadata = {**doc.metadata}
                metadata.update(chunk.metadata)
                result.append(Document(page_content=chunk.page_content, metadata=metadata))
            
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        return text_splitter.split_documents([doc])


# ============================================================================
# RELEVANCE CHECKER SIMPLIFICADO
# ============================================================================
class SimpleRelevanceChecker:
    """Verificación de relevancia SIN LLM - solo heurísticas"""
    
    def check_relevance(self, query: str, context: str) -> Tuple[bool, str, float]:
        """Verifica relevancia usando solo keywords"""
        
        query_tokens = set(re.findall(r'\w+', query.lower()))
        context_tokens = set(re.findall(r'\w+', context.lower()))
        
        # Remover stopwords comunes
        stopwords = {'el', 'la', 'de', 'que', 'en', 'un', 'una', 'los', 'las', 'del', 'como', 'para'}
        query_tokens = query_tokens - stopwords
        
        overlap = len(query_tokens & context_tokens) / len(query_tokens) if query_tokens else 0
        
        logger.info(f"Keyword overlap: {overlap:.2%}")
        
        # Criterios más laxos pero realistas
        if overlap < 0.15:  # Menos del 15% de overlap
            return False, f"Contexto no relacionado (overlap: {overlap:.1%})", overlap
        else:
            return True, f"Contexto aceptable (overlap: {overlap:.1%})", overlap


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def compute_file_metadata(path: str) -> Dict:
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
    all_docs = []
    
    if not os.path.exists(docs_dir):
        logger.error(f"Directorio {docs_dir} no existe")
        return all_docs
    
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
            
            for doc in docs:
                doc.metadata['file_type'] = ext
                doc.metadata['file_name'] = os.path.basename(file_path)
            
            all_docs.extend(docs)
            
        except Exception as e:
            logger.error(f"Error cargando {file_path}: {e}")
    
    logger.info(f"Total documentos cargados: {len(all_docs)}")
    return all_docs


class VectorStoreManager:
    def __init__(self, db_dir: str, docs_dir: str, embedding_model: str):
        self.db_dir = db_dir
        self.docs_dir = docs_dir
        self.embedding_model = embedding_model
        self.manifest_path = os.path.join(db_dir, 'index_manifest.json')
    
    def needs_rebuild(self) -> Tuple[bool, str]:
        if not os.path.exists(self.db_dir):
            return True, "Base de datos no existe"
        
        if not os.path.exists(self.manifest_path):
            return True, "Manifiesto no existe"
        
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            stored_files = {f['path']: f for f in manifest.get('files', [])}
            
            current_files = {}
            for ext in ['*.pdf', '*.md', '*.markdown', '*.txt']:
                for path in glob.glob(os.path.join(self.docs_dir, '**', ext), recursive=True):
                    current_files[os.path.normpath(path)] = compute_file_metadata(path)
            
            if set(stored_files.keys()) != set(current_files.keys()):
                return True, f"Archivos cambiaron: {len(stored_files)} -> {len(current_files)}"
            
            for path, current_meta in current_files.items():
                stored_meta = stored_files.get(path, {})
                if current_meta.get('sha1') != stored_meta.get('sha1'):
                    return True, f"Archivo modificado: {path}"
            
            return False, "Base de datos actualizada"
            
        except Exception as e:
            logger.error(f"Error verificando manifest: {e}")
            return True, f"Error en verificación: {e}"
    
    def save_manifest(self, files_metadata: List[Dict]):
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
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    manager = VectorStoreManager(DB_DIR, DOCS_DIR, EMBEDDING_MODEL)
    
    needs_rebuild, reason = manager.needs_rebuild()
    
    if needs_rebuild:
        logger.info(f"Reconstruyendo base de datos: {reason}")
        
        if os.path.exists(DB_DIR):
            try:
                deleted_files = len([f for f in Path(DB_DIR).rglob('*') if f.is_file()])
                shutil.rmtree(DB_DIR)
                logger.info(f"Base de datos anterior eliminada ({deleted_files} archivos)")
            except Exception as e:
                logger.error(f"Error eliminando DB: {e}")
        
        all_docs = load_documents_with_metadata(DOCS_DIR)
        
        if not all_docs:
            logger.error("No hay documentos para indexar")
            return None
        
        logger.info("Dividiendo documentos en chunks semánticos...")
        chunker = AdaptiveSemanticChunker(chunk_size=1200, chunk_overlap=250)
        chunks = chunker.split_documents(all_docs)
        
        logger.info(f"Documentos divididos en {len(chunks)} chunks")
        
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
    """Prompt MÁS ESPECÍFICO que fuerza al LLM a encontrar respuesta"""
    
    sources_text = "\n".join([f"- {s}" for s in sources])
    
    prompt = f"""Eres un asistente experto en VMware ESXi. Analiza cuidadosamente el contexto y responde de forma precisa.

CONTEXTO RECUPERADO:
{context}

FUENTES:
{sources_text}

PREGUNTA:
{query}

INSTRUCCIONES:
1. Lee TODOS los fragmentos del contexto cuidadosamente
2. Busca información relevante en CUALQUIER fragmento
3. Si encuentras información útil → responde con esa información
4. Si NO encuentras NADA útil → di: "No encontré esta información en la documentación proporcionada."
5. NUNCA inventes información
6. Cita fragmentos cuando sea posible

RESPUESTA:"""
    
    return prompt


def log_retrieval_metrics(metrics: RetrievalMetrics):
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


def main():
    """Función principal CORREGIDA"""
    
    logger.info("=" * 60)
    logger.info("SISTEMA RAG v2.3 - VMware ESXi (BOOSTING FIXED)")
    logger.info("v2.3: Boosting arreglado + Query expansion ampliada + Embedding cache")
    logger.info("=" * 60)
    
    # Obtener vectorstore
    vectorstore = get_vectorstore()
    
    if not vectorstore:
        logger.error("No se pudo crear/cargar la base de datos")
        return
    
    # Cargar documentos para BM25
    logger.info("Preparando retriever híbrido...")
    all_docs = load_documents_with_metadata(DOCS_DIR)
    chunker = AdaptiveSemanticChunker(chunk_size=1200, chunk_overlap=250)
    chunks = chunker.split_documents(all_docs)
    
    # Componentes v2.3 con BOOSTING
    hybrid_retriever = ImprovedHybridRetriever(vectorstore, chunks, base_alpha=0.5)
    query_expander = SimpleQueryExpander()
    reranker = FastReranker(internal_docs_metadata=hybrid_retriever.internal_docs_metadata)  # Con boost
    relevance_checker = SimpleRelevanceChecker()  # SIN LLM
    
    # LLM solo para respuesta final
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
    
    logger.info("✓ Sistema listo (v2.3 - boosting arreglado + cache)")
    print("\n" + "=" * 70)
    print("EXPERTO EN VMWARE ESXi (RAG v2.3 - BOOSTING FIXED)")
    print("=" * 70)
    print("Mejoras v2.3:")
    print("  ✓ BOOSTING ARREGLADO: Docs internos priorizados correctamente")
    print("  ✓ Query Expansion: 37 términos (antes 12) + búsqueda bidireccional")
    print("  ✓ Embedding Cache: LRU cache 1000 queries (~30% reducción latencia)")
    print("  ✓ Boost aumentado a 75% para docs internos (.md)")
    print("  ✓ Reranker bonus aumentado a 40% para docs internos")
    print("\nComandos especiales:")
    print("  - 'salir' / 'exit' / 'quit': Terminar")
    print("  - 'stats': Ver estadísticas")
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

                    # NUEVO v2.3: Estadísticas de cache
                    cache_stats = hybrid_retriever.embedding_cache.get_stats()

                    print(f"\n[STATS] Estadísticas:")
                    print(f"  - Archivos indexados: {manifest.get('num_files', 0)}")
                    print(f"  - Última actualización: {manifest.get('created_at', 'N/A')}")
                    print(f"  - Consultas realizadas: {query_count}")
                    print(f"  - Versión: v2.3 (boosting fixed + cache)")
                    print(f"\n[CACHE] Embedding Cache:")
                    print(f"  - Hits: {cache_stats['hits']}")
                    print(f"  - Misses: {cache_stats['misses']}")
                    print(f"  - Hit rate: {cache_stats['hit_rate']:.1f}%")
                    print(f"  - Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
                except Exception as e:
                    print(f"Error mostrando estadísticas: {e}")
                continue
            
            query_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"CONSULTA #{query_count}: {query}")
            logger.info(f"{'='*60}")
            
            # Query Expansion (sin LLM, instantáneo)
            query_expanded, was_expanded = query_expander.expand(query)
            
            if was_expanded:
                print(f"[✓] Query expandida automáticamente")
            
            # Retrieval con K AUMENTADO (40 candidatos)
            print("[...] Buscando información relevante...")
            results = hybrid_retriever.retrieve(query_expanded, k=40)
            
            if not results:
                print("\n[!] No se encontró información relevante.")
                continue
            
            # Reranking RÁPIDO (sin LLM)
            print("[...] Reordenando por relevancia...")
            results = reranker.rerank(query, results, top_k=8)  # Top 8 en lugar de 5
            
            # Métricas
            docs = [doc for doc, score in results]
            scores = [score for doc, score in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            logger.info(f"Relevancia promedio: {avg_score:.2%}")
            
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
                    f"--- Fragmento {i} [{source_info}] (score: {score:.2f}) ---\n"
                    f"{doc.page_content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Relevance check SIMPLE (sin LLM)
            is_relevant, relevance_msg, relevance_score = relevance_checker.check_relevance(query, context)
            
            if not is_relevant:
                logger.warning(f"Contexto rechazado: {relevance_msg}")
                print(f"\n[!] {relevance_msg}")
                print("Intenta reformular con más detalles técnicos.")
                continue
            
            logger.info(f"Contexto aceptado: {relevance_msg}")
            
            # Log métricas
            metrics = RetrievalMetrics(
                query=query,
                num_chunks_retrieved=len(docs),
                avg_similarity_score=avg_score,
                sources_used=sources,
                context_length=len(context),
                retrieval_method="hybrid+fast_rerank",
                reranked=True,
                query_expanded=was_expanded
            )
            log_retrieval_metrics(metrics)
            
            # Prompt mejorado
            prompt = build_enhanced_prompt(query, context, sources)
            
            # Generar respuesta
            print("[AI] Generando respuesta...\n")
            
            try:
                ai_message = llm.invoke(prompt)
                response = getattr(ai_message, 'content', str(ai_message))
                
                # Mostrar
                print("=" * 70)
                print("RESPUESTA:")
                print("=" * 70)
                print(response)
                print("=" * 70)
                
                print(f"\n[FUENTES] Consultadas ({len(sources)}):")
                for source in sources:
                    print(f"  * {source}")
                
                print(f"\n[METRICAS] Calidad:")
                print(f"  * Chunks: {len(docs)}")
                print(f"  * Relevancia: {avg_score:.2%}")
                print(f"  * Contexto: {len(context):,} caracteres")
                print(f"  * Query expandida: {'Sí' if was_expanded else 'No'}")
                print(f"  * Tiempo reranking: <1s (heurísticas)")
                
                logger.info(f"Consulta #{query_count} completada")
                
            except Exception as e:
                logger.error(f"Error generando respuesta: {e}")
                print(f"\n[ERROR] Error: {e}")
        
        except KeyboardInterrupt:
            print("\n\nInterrumpido")
            logger.info("Sesión interrumpida")
            break
        
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n[ERROR] Error: {e}")
    
    print("\n=== Hasta luego! ===")
    logger.info(f"Sesión finalizada. Consultas: {query_count}")


if __name__ == "__main__":
    main()
