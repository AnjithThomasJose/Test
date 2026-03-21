"""
Resume Loader - End-to-End Ingestion Orchestrator

This module orchestrates the complete resume ingestion pipeline:
1. Parse resume (reuse groq_resume_parser)
2. Extract metadata tags (smart_tagger)
3. Chunk resume (chunker)
4. Generate embeddings
5. Store in ChromaDB

This is the main entry point for resume ingestion.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .smart_tagger import SmartTagger, ResumeMetadata
from .chunker import ResumeChunker, ResumeChunk

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Result of resume ingestion process.
    
    Attributes:
        success: Whether ingestion succeeded
        candidate_id: Candidate UID
        chunks_created: Number of chunks created
        metadata: Extracted metadata
        error: Error message if failed
        processing_time_ms: Time taken (milliseconds)
    """
    success: bool
    candidate_id: str
    chunks_created: int
    metadata: Optional[ResumeMetadata]
    error: Optional[str] = None
    processing_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "success": self.success,
            "candidate_id": self.candidate_id,
            "chunks_created": self.chunks_created,
            "processing_time_ms": self.processing_time_ms
        }
        
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        
        if self.error:
            result["error"] = self.error
        
        return result


class ResumeLoader:
    """
    End-to-end resume ingestion orchestrator.
    
    This class manages the complete pipeline:
    1. Resume parsing (delegates to groq_resume_parser)
    2. Metadata extraction (smart_tagger)
    3. Semantic chunking (chunker)
    4. Embedding generation (ChromaDB)
    5. Storage (ChromaDB)
    
    Usage:
        loader = ResumeLoader()
        
        # From URL
        result = await loader.load_from_url(
            candidate_id="uid_123",
            resume_url="https://example.com/resume.pdf"
        )
        
        # From structured resume
        result = await loader.load_from_structured(
            candidate_id="uid_123",
            structured_resume=parsed_resume
        )
    """
    
    def __init__(self, collection_name: str = "candidates_v1"):
        """
        Initialize resume loader.
        
        Args:
            collection_name: ChromaDB collection to store in
        """
        self.collection_name = collection_name
        self.smart_tagger = SmartTagger()
        self.chunker = ResumeChunker()
        self._chroma_collection = None
        
        log.info(f"ResumeLoader initialized for collection: {collection_name}")
    
    def _get_collection(self):
        """Get or create ChromaDB collection"""
        if self._chroma_collection is None:
            try:
                import chroma
                self._chroma_collection = chroma._get_collection(self.collection_name)
                log.info(f"Connected to collection: {self.collection_name}")
            except Exception as e:
                log.error(f"Failed to get collection: {e}")
                raise
        return self._chroma_collection
    
    async def load_from_url(
        self,
        candidate_id: str,
        resume_url: str,
        use_llm_tagging: bool = True
    ) -> IngestionResult:
        """
        Load resume from URL (complete pipeline).
        
        Args:
            candidate_id: Unique candidate identifier
            resume_url: URL to resume PDF/document
            use_llm_tagging: Whether to use LLM for metadata extraction
            
        Returns:
            IngestionResult with success status and details
        """
        import time
        start_time = time.time()
        
        try:
            log.info(f"Starting resume ingestion for candidate: {candidate_id}")
            log.info(f"Resume URL: {resume_url}")
            
            # Step 1: Parse resume using existing groq_resume_parser
            log.info("Step 1: Parsing resume...")
            structured_resume = await self._parse_resume(resume_url)
            
            if not structured_resume:
                return IngestionResult(
                    success=False,
                    candidate_id=candidate_id,
                    chunks_created=0,
                    metadata=None,
                    error="Failed to parse resume",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Step 2-5: Process structured resume
            result = await self.load_from_structured(
                candidate_id=candidate_id,
                structured_resume=structured_resume,
                use_llm_tagging=use_llm_tagging
            )
            
            return result
            
        except Exception as e:
            log.error(f"Resume ingestion failed: {e}", exc_info=True)
            return IngestionResult(
                success=False,
                candidate_id=candidate_id,
                chunks_created=0,
                metadata=None,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def load_from_structured(
        self,
        candidate_id: str,
        structured_resume: Dict[str, Any],
        use_llm_tagging: bool = True
    ) -> IngestionResult:
        """
        Load resume from structured data (steps 2-5).
        
        Args:
            candidate_id: Unique candidate identifier
            structured_resume: Parsed resume data from groq_resume_parser
            use_llm_tagging: Whether to use LLM for metadata extraction
            
        Returns:
            IngestionResult with success status and details
        """
        import time
        start_time = time.time()
        
        try:
            log.info(f"Processing structured resume for candidate: {candidate_id}")
            
            # Step 2: Extract metadata tags
            log.info("Step 2: Extracting metadata tags...")
            if use_llm_tagging:
                try:
                    metadata = await self.smart_tagger.extract_metadata(structured_resume)
                except Exception as e:
                    log.warning(f"LLM tagging failed, using fallback: {e}")
                    metadata = self.smart_tagger.extract_metadata_fallback(structured_resume)
            else:
                metadata = self.smart_tagger.extract_metadata_fallback(structured_resume)
            
            log.info(
                f"Metadata extracted: {len(metadata.skills)} skills, "
                f"{metadata.total_years_exp} years, {metadata.seniority_level}"
            )
            
            # Step 3: Chunk resume
            log.info("Step 3: Chunking resume...")
            chunks = self.chunker.chunk_resume(structured_resume)
            log.info(f"Created {len(chunks)} chunks")
            
            # Step 4 & 5: Generate embeddings and store in ChromaDB
            log.info("Step 4-5: Generating embeddings and storing...")
            await self._store_chunks(candidate_id, chunks, metadata, structured_resume)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            log.info(
                f"✅ Resume ingestion complete: {candidate_id}, "
                f"{len(chunks)} chunks, {processing_time_ms}ms"
            )
            
            return IngestionResult(
                success=True,
                candidate_id=candidate_id,
                chunks_created=len(chunks),
                metadata=metadata,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            log.error(f"Structured resume processing failed: {e}", exc_info=True)
            return IngestionResult(
                success=False,
                candidate_id=candidate_id,
                chunks_created=0,
                metadata=None,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _parse_resume(self, resume_url: str) -> Optional[Dict[str, Any]]:
        """
        Parse resume using existing groq_resume_parser.
        
        Args:
            resume_url: URL to resume document
            
        Returns:
            Structured resume data or None if parsing fails
        """
        try:
            # Import groq_resume_parser
            from agents.groq_resume_parser import groq_resume_parser_agent
            
            # Create state for parser
            state = {
                "resume_url": resume_url,
                "tenant_id": "default_tenant"
            }
            
            # Parse resume
            result = await groq_resume_parser_agent(state)
            
            # Extract structured resume
            if result and isinstance(result, dict):
                structured_resume = result.get("structured_resume")
                if structured_resume:
                    log.info("Resume parsed successfully")
                    return structured_resume
            
            log.error("Failed to extract structured_resume from parser result")
            return None
            
        except Exception as e:
            log.error(f"Resume parsing failed: {e}", exc_info=True)
            return None
    
    async def _store_chunks(
        self,
        candidate_id: str,
        chunks: List[ResumeChunk],
        metadata: ResumeMetadata,
        structured_resume: Optional[Dict[str, Any]] = None
    ):
        """
        Store chunks in ChromaDB with embeddings.
        
        Args:
            candidate_id: Candidate UID
            chunks: List of resume chunks
            metadata: Extracted metadata
            structured_resume: Optional full structured resume to extract name from
        """
        try:
            collection = self._get_collection()
            
            # Extract candidate name from structured resume if available
            candidate_name = ""
            if structured_resume:
                name_value = structured_resume.get("Name") or structured_resume.get("name")
                if isinstance(name_value, list) and len(name_value) > 0:
                    candidate_name = name_value[0]
                elif isinstance(name_value, str):
                    candidate_name = name_value
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Create unique ID for each chunk
                chunk_id = f"{candidate_id}_chunk_{chunk.chunk_index}"
                ids.append(chunk_id)
                
                # Document is the chunk content
                documents.append(chunk.content)
                
                # Metadata combines chunk metadata + extracted metadata + name
                chunk_metadata = {
                    "candidate_id": candidate_id,
                    "section_type": chunk.section_type,
                    "chunk_index": chunk.chunk_index,
                    "name": candidate_name,  # Add candidate name to chunk metadata
                    **metadata.to_dict(),  # Add searchable metadata
                    **chunk.metadata  # Add chunk-specific metadata
                }
                
                # Normalize metadata for ChromaDB
                import chroma
                normalized_metadata = chroma.normalize_metadata(chunk_metadata)
                metadatas.append(normalized_metadata)
            
            # Upsert to ChromaDB (embeddings generated automatically)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            )
            
            log.info(f"Stored {len(chunks)} chunks for candidate {candidate_id}")
            
        except Exception as e:
            log.error(f"Failed to store chunks: {e}", exc_info=True)
            raise
    
    async def delete_candidate(self, candidate_id: str) -> bool:
        """
        Delete all chunks for a candidate.
        
        Args:
            candidate_id: Candidate UID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self._get_collection()
            
            # Delete all chunks for this candidate
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.delete(
                    where={"candidate_id": candidate_id}
                )
            )
            
            log.info(f"Deleted all chunks for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to delete candidate: {e}")
            return False
    
    async def get_candidate_chunks(self, candidate_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a candidate.
        
        Args:
            candidate_id: Candidate UID
            
        Returns:
            List of chunk dictionaries
        """
        try:
            collection = self._get_collection()
            
            # Get all chunks for this candidate
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.get(
                    where={"candidate_id": candidate_id},
                    include=['documents', 'metadatas']
                )
            )
            
            chunks = []
            if results and results.get('ids'):
                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        "id": chunk_id,
                        "document": results['documents'][i] if i < len(results['documents']) else None,
                        "metadata": results['metadatas'][i] if i < len(results['metadatas']) else {}
                    })
            
            log.info(f"Retrieved {len(chunks)} chunks for candidate {candidate_id}")
            return chunks
            
        except Exception as e:
            log.error(f"Failed to retrieve chunks: {e}")
            return []


# Convenience function
async def load_resume(
    candidate_id: str,
    resume_url: Optional[str] = None,
    structured_resume: Optional[Dict[str, Any]] = None,
    collection_name: str = "candidates_v1",
    use_llm_tagging: bool = True
) -> IngestionResult:
    """
    Load resume into ChromaDB (convenience function).
    
    Args:
        candidate_id: Unique candidate identifier
        resume_url: URL to resume (if loading from URL)
        structured_resume: Parsed resume data (if already parsed)
        collection_name: ChromaDB collection name
        use_llm_tagging: Whether to use LLM for metadata extraction
        
    Returns:
        IngestionResult with success status
    """
    loader = ResumeLoader(collection_name=collection_name)
    
    if resume_url:
        return await loader.load_from_url(candidate_id, resume_url, use_llm_tagging)
    elif structured_resume:
        return await loader.load_from_structured(candidate_id, structured_resume, use_llm_tagging)
    else:
        raise ValueError("Either resume_url or structured_resume must be provided")
