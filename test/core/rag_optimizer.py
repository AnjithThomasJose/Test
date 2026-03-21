"""
RAG optimization utilities with MMR, token targeting, and extractive compression.
Provides optimized retrieval with k=3-5, target 1200-1500 tokens, and compression.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import re

@dataclass
class RAGConfig:
    """Configuration for RAG retrieval."""
    k: int = 5  # Number of documents to retrieve
    target_tokens: int = 1200  # Target context tokens (1200-1500)
    use_mmr: bool = True  # Use Maximal Marginal Relevance
    mmr_diversity: float = 0.5  # MMR diversity parameter (0.0 = relevance only, 1.0 = diversity only)
    enable_compression: bool = True  # Enable extractive compression
    min_tokens_per_doc: int = 50  # Minimum tokens per document after compression

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
    return len(text) // 4

def extractive_compress(text: str, target_tokens: int) -> str:
    """
    Compress text extractively to target token count.
    Preserves sentences and key information.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Estimate tokens per sentence
    sentence_tokens = [estimate_tokens(s) for s in sentences]
    
    # Greedily select sentences up to target
    selected = []
    current_tokens = 0
    
    for sentence, tokens in zip(sentences, sentence_tokens):
        if current_tokens + tokens <= target_tokens:
            selected.append(sentence)
            current_tokens += tokens
        else:
            # Try to fit partial sentence if close
            remaining = target_tokens - current_tokens
            if remaining > 20:  # At least 20 tokens remaining
                # Take first part of sentence
                partial = sentence[:remaining * 4]  # Approximate character count
                selected.append(partial + "...")
            break
    
    return ". ".join(selected) + "."

def mmr_rerank(
    documents: List[Dict[str, Any]],
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    diversity: float = 0.5,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance and diversity:
    - High diversity (1.0): Maximize diversity
    - Low diversity (0.0): Maximize relevance only
    
    Args:
        documents: List of document dicts
        query_embedding: Query embedding vector
        document_embeddings: List of document embedding vectors
        diversity: Diversity parameter (0.0-1.0)
        k: Number of documents to return
    
    Returns:
        Reranked list of documents
    """
    if not documents or not document_embeddings:
        return documents[:k]
    
    # Simple cosine similarity (for production, use proper vector similarity)
    def cosine_sim(a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    selected = []
    remaining = list(range(len(documents)))
    
    # Select first document (most relevant)
    if remaining:
        similarities = [cosine_sim(query_embedding, doc_emb) for doc_emb in document_embeddings]
        first_idx = max(remaining, key=lambda i: similarities[i])
        selected.append(first_idx)
        remaining.remove(first_idx)
    
    # Select remaining documents using MMR
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, document_embeddings[idx])
            
            # Diversity from already selected
            max_similarity = 0.0
            for sel_idx in selected:
                similarity = cosine_sim(document_embeddings[idx], document_embeddings[sel_idx])
                max_similarity = max(max_similarity, similarity)
            
            # MMR score
            mmr_score = diversity * relevance - (1 - diversity) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break
    
    return [documents[i] for i in selected]

async def retrieve_compress(
    query: str,
    retriever: Callable,
    embedder: Optional[Callable] = None,
    config: Optional[RAGConfig] = None
) -> Dict[str, Any]:
    """
    Retrieve and compress documents for RAG.
    
    Args:
        query: Query string
        retriever: Function that retrieves documents (returns List[Dict])
        embedder: Optional function to create embeddings
        config: RAG configuration
    
    Returns:
        Dictionary with compressed_context, documents, and metadata
    """
    if config is None:
        config = RAGConfig()
    
    # Retrieve initial documents
    documents = await retriever(query, k=config.k * 2)  # Retrieve more for MMR
    
    if not documents:
        return {
            "compressed_context": "",
            "documents": [],
            "metadata": {
                "k": 0,
                "total_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 0.0
            }
        }
    
    # Apply MMR if enabled and embeddings available
    if config.use_mmr and embedder:
        query_embedding = await embedder(query)
        doc_embeddings = [await embedder(doc.get("content", doc.get("text", ""))) for doc in documents]
        documents = mmr_rerank(
            documents,
            query_embedding,
            doc_embeddings,
            config.mmr_diversity,
            config.k
        )
    else:
        documents = documents[:config.k]
    
    # Combine documents
    combined_text = "\n\n".join(
        doc.get("content", doc.get("text", str(doc)))
        for doc in documents
    )
    
    total_tokens = estimate_tokens(combined_text)
    
    # Compress if needed
    if config.enable_compression and total_tokens > config.target_tokens:
        # Distribute target tokens across documents
        tokens_per_doc = max(
            config.min_tokens_per_doc,
            config.target_tokens // len(documents)
        )
        
        compressed_docs = []
        for doc in documents:
            doc_text = doc.get("content", doc.get("text", str(doc)))
            compressed = extractive_compress(doc_text, tokens_per_doc)
            compressed_docs.append(compressed)
        
        compressed_context = "\n\n".join(compressed_docs)
        compressed_tokens = estimate_tokens(compressed_context)
    else:
        compressed_context = combined_text
        compressed_tokens = total_tokens
    
    return {
        "compressed_context": compressed_context,
        "documents": documents,
        "metadata": {
            "k": len(documents),
            "total_tokens": total_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compressed_tokens / total_tokens if total_tokens > 0 else 0.0,
            "used_mmr": config.use_mmr and embedder is not None
        }
    }






