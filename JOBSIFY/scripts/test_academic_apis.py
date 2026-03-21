#!/usr/bin/env python3
"""
Test script for academic API adapters

Usage:
    python scripts/test_academic_apis.py --topic "machine learning" --source arxiv
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.academic_api_adapters import AcademicAPIManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Test academic API adapters"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to search for"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "arxiv", "google_scholar", "ieee", "pubmed", "google_books", "open_library", "crossref", "wikibooks"],
        default="all",
        help="Academic source to search (default: all)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["paper", "book", "both"],
        default="both",
        help="Material type to search (default: both)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per source (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        manager = AcademicAPIManager()
        
        logger.info("="*60)
        logger.info("🔬 ACADEMIC API TEST")
        logger.info("="*60)
        logger.info(f"Topic: {args.topic}")
        logger.info(f"Source: {args.source}")
        logger.info(f"Type: {args.type}")
        logger.info(f"Max results: {args.max_results}")
        logger.info("")
        
        sources = None if args.source == "all" else [args.source]
        
        all_results = []
        
        # Search for papers
        if args.type in ["paper", "both"]:
            papers = manager.search_all_sources(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results,
                material_type="paper"
            )
            all_results.extend(papers)
        
        # Search for books
        if args.type in ["book", "both"]:
            books = manager.search_books(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results
            )
            all_results.extend(books)
        
        if not all_results:
            logger.warning("⚠️  No materials found")
            return 1
        
        # Separate papers and books
        papers = [r for r in all_results if r.get('type') == 'paper']
        books = [r for r in all_results if r.get('type') == 'book']
        
        if papers:
            logger.info(f"\n✅ Found {len(papers)} papers\n")
            logger.info("="*60)
            logger.info("📄 PAPERS")
            logger.info("="*60)
            
            for i, paper in enumerate(papers, 1):
                logger.info(f"\n[{i}] {paper.get('title', 'No title')}")
                logger.info(f"    Source: {paper.get('source', 'Unknown')}")
                logger.info(f"    URL: {paper.get('url', 'No URL')}")
                if paper.get('authors'):
                    logger.info(f"    Authors: {', '.join(paper['authors'][:3])}")
                    if len(paper['authors']) > 3:
                        logger.info(f"             ... and {len(paper['authors']) - 3} more")
                if paper.get('year'):
                    logger.info(f"    Year: {paper['year']}")
                if paper.get('citations'):
                    logger.info(f"    Citations: {paper['citations']}")
                if paper.get('abstract'):
                    abstract = paper['abstract'][:200]
                    logger.info(f"    Abstract: {abstract}...")
        
        if books:
            logger.info(f"\n✅ Found {len(books)} books\n")
            logger.info("="*60)
            logger.info("📖 BOOKS")
            logger.info("="*60)
            
            for i, book in enumerate(books, 1):
                logger.info(f"\n[{i}] {book.get('title', 'No title')}")
                logger.info(f"    Source: {book.get('source', 'Unknown')}")
                logger.info(f"    URL: {book.get('url', 'No URL')}")
                if book.get('authors'):
                    logger.info(f"    Authors: {', '.join(book['authors'][:3])}")
                    if len(book['authors']) > 3:
                        logger.info(f"             ... and {len(book['authors']) - 3} more")
                if book.get('year'):
                    logger.info(f"    Year: {book['year']}")
                if book.get('publisher'):
                    logger.info(f"    Publisher: {book['publisher']}")
                if book.get('isbn'):
                    logger.info(f"    ISBN: {book['isbn']}")
                if book.get('description'):
                    desc = book['description'][:200]
                    logger.info(f"    Description: {desc}...")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Test complete!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Test script for academic API adapters

Usage:
    python scripts/test_academic_apis.py --topic "machine learning" --source arxiv
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.academic_api_adapters import AcademicAPIManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Test academic API adapters"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to search for"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "arxiv", "google_scholar", "ieee", "pubmed", "google_books", "open_library", "crossref", "wikibooks"],
        default="all",
        help="Academic source to search (default: all)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["paper", "book", "both"],
        default="both",
        help="Material type to search (default: both)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per source (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        manager = AcademicAPIManager()
        
        logger.info("="*60)
        logger.info("🔬 ACADEMIC API TEST")
        logger.info("="*60)
        logger.info(f"Topic: {args.topic}")
        logger.info(f"Source: {args.source}")
        logger.info(f"Type: {args.type}")
        logger.info(f"Max results: {args.max_results}")
        logger.info("")
        
        sources = None if args.source == "all" else [args.source]
        
        all_results = []
        
        # Search for papers
        if args.type in ["paper", "both"]:
            papers = manager.search_all_sources(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results,
                material_type="paper"
            )
            all_results.extend(papers)
        
        # Search for books
        if args.type in ["book", "both"]:
            books = manager.search_books(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results
            )
            all_results.extend(books)
        
        if not all_results:
            logger.warning("⚠️  No materials found")
            return 1
        
        # Separate papers and books
        papers = [r for r in all_results if r.get('type') == 'paper']
        books = [r for r in all_results if r.get('type') == 'book']
        
        if papers:
            logger.info(f"\n✅ Found {len(papers)} papers\n")
            logger.info("="*60)
            logger.info("📄 PAPERS")
            logger.info("="*60)
            
            for i, paper in enumerate(papers, 1):
                logger.info(f"\n[{i}] {paper.get('title', 'No title')}")
                logger.info(f"    Source: {paper.get('source', 'Unknown')}")
                logger.info(f"    URL: {paper.get('url', 'No URL')}")
                if paper.get('authors'):
                    logger.info(f"    Authors: {', '.join(paper['authors'][:3])}")
                    if len(paper['authors']) > 3:
                        logger.info(f"             ... and {len(paper['authors']) - 3} more")
                if paper.get('year'):
                    logger.info(f"    Year: {paper['year']}")
                if paper.get('citations'):
                    logger.info(f"    Citations: {paper['citations']}")
                if paper.get('abstract'):
                    abstract = paper['abstract'][:200]
                    logger.info(f"    Abstract: {abstract}...")
        
        if books:
            logger.info(f"\n✅ Found {len(books)} books\n")
            logger.info("="*60)
            logger.info("📖 BOOKS")
            logger.info("="*60)
            
            for i, book in enumerate(books, 1):
                logger.info(f"\n[{i}] {book.get('title', 'No title')}")
                logger.info(f"    Source: {book.get('source', 'Unknown')}")
                logger.info(f"    URL: {book.get('url', 'No URL')}")
                if book.get('authors'):
                    logger.info(f"    Authors: {', '.join(book['authors'][:3])}")
                    if len(book['authors']) > 3:
                        logger.info(f"             ... and {len(book['authors']) - 3} more")
                if book.get('year'):
                    logger.info(f"    Year: {book['year']}")
                if book.get('publisher'):
                    logger.info(f"    Publisher: {book['publisher']}")
                if book.get('isbn'):
                    logger.info(f"    ISBN: {book['isbn']}")
                if book.get('description'):
                    desc = book['description'][:200]
                    logger.info(f"    Description: {desc}...")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Test complete!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Test script for academic API adapters

Usage:
    python scripts/test_academic_apis.py --topic "machine learning" --source arxiv
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.academic_api_adapters import AcademicAPIManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Test academic API adapters"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to search for"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "arxiv", "google_scholar", "ieee", "pubmed", "google_books", "open_library", "crossref", "wikibooks"],
        default="all",
        help="Academic source to search (default: all)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["paper", "book", "both"],
        default="both",
        help="Material type to search (default: both)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per source (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        manager = AcademicAPIManager()
        
        logger.info("="*60)
        logger.info("🔬 ACADEMIC API TEST")
        logger.info("="*60)
        logger.info(f"Topic: {args.topic}")
        logger.info(f"Source: {args.source}")
        logger.info(f"Type: {args.type}")
        logger.info(f"Max results: {args.max_results}")
        logger.info("")
        
        sources = None if args.source == "all" else [args.source]
        
        all_results = []
        
        # Search for papers
        if args.type in ["paper", "both"]:
            papers = manager.search_all_sources(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results,
                material_type="paper"
            )
            all_results.extend(papers)
        
        # Search for books
        if args.type in ["book", "both"]:
            books = manager.search_books(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results
            )
            all_results.extend(books)
        
        if not all_results:
            logger.warning("⚠️  No materials found")
            return 1
        
        # Separate papers and books
        papers = [r for r in all_results if r.get('type') == 'paper']
        books = [r for r in all_results if r.get('type') == 'book']
        
        if papers:
            logger.info(f"\n✅ Found {len(papers)} papers\n")
            logger.info("="*60)
            logger.info("📄 PAPERS")
            logger.info("="*60)
            
            for i, paper in enumerate(papers, 1):
                logger.info(f"\n[{i}] {paper.get('title', 'No title')}")
                logger.info(f"    Source: {paper.get('source', 'Unknown')}")
                logger.info(f"    URL: {paper.get('url', 'No URL')}")
                if paper.get('authors'):
                    logger.info(f"    Authors: {', '.join(paper['authors'][:3])}")
                    if len(paper['authors']) > 3:
                        logger.info(f"             ... and {len(paper['authors']) - 3} more")
                if paper.get('year'):
                    logger.info(f"    Year: {paper['year']}")
                if paper.get('citations'):
                    logger.info(f"    Citations: {paper['citations']}")
                if paper.get('abstract'):
                    abstract = paper['abstract'][:200]
                    logger.info(f"    Abstract: {abstract}...")
        
        if books:
            logger.info(f"\n✅ Found {len(books)} books\n")
            logger.info("="*60)
            logger.info("📖 BOOKS")
            logger.info("="*60)
            
            for i, book in enumerate(books, 1):
                logger.info(f"\n[{i}] {book.get('title', 'No title')}")
                logger.info(f"    Source: {book.get('source', 'Unknown')}")
                logger.info(f"    URL: {book.get('url', 'No URL')}")
                if book.get('authors'):
                    logger.info(f"    Authors: {', '.join(book['authors'][:3])}")
                    if len(book['authors']) > 3:
                        logger.info(f"             ... and {len(book['authors']) - 3} more")
                if book.get('year'):
                    logger.info(f"    Year: {book['year']}")
                if book.get('publisher'):
                    logger.info(f"    Publisher: {book['publisher']}")
                if book.get('isbn'):
                    logger.info(f"    ISBN: {book['isbn']}")
                if book.get('description'):
                    desc = book['description'][:200]
                    logger.info(f"    Description: {desc}...")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Test complete!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Test script for academic API adapters

Usage:
    python scripts/test_academic_apis.py --topic "machine learning" --source arxiv
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.academic_api_adapters import AcademicAPIManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Test academic API adapters"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to search for"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "arxiv", "google_scholar", "ieee", "pubmed", "google_books", "open_library", "crossref", "wikibooks"],
        default="all",
        help="Academic source to search (default: all)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["paper", "book", "both"],
        default="both",
        help="Material type to search (default: both)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per source (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        manager = AcademicAPIManager()
        
        logger.info("="*60)
        logger.info("🔬 ACADEMIC API TEST")
        logger.info("="*60)
        logger.info(f"Topic: {args.topic}")
        logger.info(f"Source: {args.source}")
        logger.info(f"Type: {args.type}")
        logger.info(f"Max results: {args.max_results}")
        logger.info("")
        
        sources = None if args.source == "all" else [args.source]
        
        all_results = []
        
        # Search for papers
        if args.type in ["paper", "both"]:
            papers = manager.search_all_sources(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results,
                material_type="paper"
            )
            all_results.extend(papers)
        
        # Search for books
        if args.type in ["book", "both"]:
            books = manager.search_books(
                query=args.topic,
                sources=sources,
                max_results_per_source=args.max_results
            )
            all_results.extend(books)
        
        if not all_results:
            logger.warning("⚠️  No materials found")
            return 1
        
        # Separate papers and books
        papers = [r for r in all_results if r.get('type') == 'paper']
        books = [r for r in all_results if r.get('type') == 'book']
        
        if papers:
            logger.info(f"\n✅ Found {len(papers)} papers\n")
            logger.info("="*60)
            logger.info("📄 PAPERS")
            logger.info("="*60)
            
            for i, paper in enumerate(papers, 1):
                logger.info(f"\n[{i}] {paper.get('title', 'No title')}")
                logger.info(f"    Source: {paper.get('source', 'Unknown')}")
                logger.info(f"    URL: {paper.get('url', 'No URL')}")
                if paper.get('authors'):
                    logger.info(f"    Authors: {', '.join(paper['authors'][:3])}")
                    if len(paper['authors']) > 3:
                        logger.info(f"             ... and {len(paper['authors']) - 3} more")
                if paper.get('year'):
                    logger.info(f"    Year: {paper['year']}")
                if paper.get('citations'):
                    logger.info(f"    Citations: {paper['citations']}")
                if paper.get('abstract'):
                    abstract = paper['abstract'][:200]
                    logger.info(f"    Abstract: {abstract}...")
        
        if books:
            logger.info(f"\n✅ Found {len(books)} books\n")
            logger.info("="*60)
            logger.info("📖 BOOKS")
            logger.info("="*60)
            
            for i, book in enumerate(books, 1):
                logger.info(f"\n[{i}] {book.get('title', 'No title')}")
                logger.info(f"    Source: {book.get('source', 'Unknown')}")
                logger.info(f"    URL: {book.get('url', 'No URL')}")
                if book.get('authors'):
                    logger.info(f"    Authors: {', '.join(book['authors'][:3])}")
                    if len(book['authors']) > 3:
                        logger.info(f"             ... and {len(book['authors']) - 3} more")
                if book.get('year'):
                    logger.info(f"    Year: {book['year']}")
                if book.get('publisher'):
                    logger.info(f"    Publisher: {book['publisher']}")
                if book.get('isbn'):
                    logger.info(f"    ISBN: {book['isbn']}")
                if book.get('description'):
                    desc = book['description'][:200]
                    logger.info(f"    Description: {desc}...")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Test complete!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

