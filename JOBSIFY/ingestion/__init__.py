"""
Ingestion Pipeline - Data Processing Layer

This module handles the end-to-end process of ingesting resumes and storing
them in ChromaDB with proper embeddings and metadata.

Key Components:
- resume_loader: End-to-end resume ingestion orchestrator
- smart_tagger: LLM-based metadata extraction
- chunker: Semantic resume chunking

Design Principles:
- Reuses Existing Infrastructure: Leverages groq_resume_parser.py
- Smart Metadata: LLM extracts searchable tags (skills, experience, location)
- Semantic Chunking: Splits resumes by sections (Work, Education, Skills)
"""

from .resume_loader import ResumeLoader, load_resume
from .smart_tagger import SmartTagger, extract_metadata_tags
from .chunker import ResumeChunker, chunk_resume

__all__ = [
    "ResumeLoader",
    "load_resume",
    "SmartTagger",
    "extract_metadata_tags",
    "ResumeChunker",
    "chunk_resume"
]

__version__ = "1.0.0"
