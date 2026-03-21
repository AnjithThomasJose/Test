"""
Academic API Adapters for Research Papers, Books, and Academic Materials

Supports:
- Google Scholar (via SerpAPI)
- arXiv API
- IEEE Xplore (via web scraping/SerpAPI)
- PubMed (for medical/health sciences)
- Google Books API
- Open Library API
- Crossref API
- Wikibooks API
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import httpx
from datetime import datetime

from core.http_client import get_http_client

log = logging.getLogger(__name__)


class AcademicAPIAdapter(ABC):
    """Base class for academic API adapters"""
    
    @abstractmethod
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for academic papers"""
        pass
    
    @abstractmethod
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize paper data to standard format"""
        pass
    
    async def search_books(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for books (optional, for book-specific adapters)"""
        return []
    
    def normalize_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize book data to standard format (optional)"""
        return book_data


class GoogleScholarAdapter(AcademicAPIAdapter):
    """
    Google Scholar adapter - OPTIONAL (requires paid SerpAPI)
    
    Note: Google Scholar doesn't have an official free API.
    This adapter only works if SERPAPI_API_KEY is set.
    Use arXiv, PubMed, or Crossref instead (all free).
    """
    
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        self.base_url = "https://serpapi.com/search"
        
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google Scholar for papers - only if SerpAPI key is available"""
        if not self.api_key:
            log.debug("Google Scholar requires SerpAPI (paid). Use arXiv, PubMed, or Crossref instead (all free).")
            return []
        
        try:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 20)  # SerpAPI limit
            }
            
            http_client = await get_http_client()
            response = await http_client.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results[:max_results]:
                normalized = self.normalize_paper(result)
                if normalized:
                    papers.append(normalized)
            
            log.info(f"Found {len(papers)} papers from Google Scholar for '{query}'")
            return papers
            
        except Exception as e:
            log.error(f"Error searching Google Scholar: {e}")
            return []
        
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Google Scholar result to standard format"""
        try:
            title = paper_data.get("title", "")
            link = paper_data.get("link", "")
            
            if not title or not link:
                return None
            
            # Extract authors
            authors = []
            if "publication_info" in paper_data:
                authors_str = paper_data["publication_info"].get("authors", [])
                if isinstance(authors_str, list):
                    authors = [a.get("name", "") for a in authors_str if isinstance(a, dict)]
                elif isinstance(authors_str, str):
                    authors = [a.strip() for a in authors_str.split(",")]
            
            # Extract year
            year = None
            if "publication_info" in paper_data:
                year_str = paper_data["publication_info"].get("year", "")
                try:
                    year = int(year_str) if year_str else None
                except (ValueError, TypeError):
                    pass
            
            # Extract snippet/abstract
            snippet = paper_data.get("snippet", "")
            
            # Extract citation count
            citations = paper_data.get("inline_links", {}).get("cited_by", {}).get("total", 0)
            
            return {
                "title": title,
                "url": link,
                "authors": authors,
                "year": year,
                "abstract": snippet,
                "citations": citations,
                "source": "Google Scholar",
                "type": "paper"
            }
        except Exception as e:
            log.warning(f"Error normalizing Google Scholar paper: {e}")
            return None


class ArxivAdapter(AcademicAPIAdapter):
    """
    arXiv API adapter
    
    arXiv provides a free, official API for searching papers.
    """
    
    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"  # Fixed: Use HTTPS instead of HTTP
        
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers"""
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(max_results, 100),  # arXiv limit
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            http_client = await get_http_client()
            response = await http_client.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            
            # Namespace handling
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            entries = root.findall('atom:entry', ns)
            
            for entry in entries[:max_results]:
                normalized = self._parse_arxiv_entry(entry, ns)
                if normalized:
                    papers.append(normalized)
            
            log.info(f"Found {len(papers)} papers from arXiv for '{query}'")
            return papers
            
        except Exception as e:
            log.error(f"Error searching arXiv: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry, ns) -> Optional[Dict[str, Any]]:
        """Parse a single arXiv entry"""
        try:
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            link_elem = entry.find('atom:id', ns)
            url = link_elem.text if link_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract published date
            published_elem = entry.find('atom:published', ns)
            year = None
            if published_elem is not None:
                try:
                    date_str = published_elem.text
                    year = int(date_str[:4]) if len(date_str) >= 4 else None
                except (ValueError, TypeError, AttributeError):
                    pass
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term', '')
                if term:
                    categories.append(term)
            
            if not title or not url:
                return None
            
            return {
                "title": title,
                "url": url,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "categories": categories,
                "source": "arXiv",
                "type": "paper"
            }
        except Exception as e:
            log.warning(f"Error parsing arXiv entry: {e}")
            return None
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize arXiv paper (already normalized in _parse_arxiv_entry)"""
        return paper_data


class IEEEXploreAdapter(AcademicAPIAdapter):
    """
    IEEE Xplore adapter - OPTIONAL (requires paid SerpAPI)
    
    Note: IEEE Xplore requires subscription for official API.
    This adapter only works if SERPAPI_API_KEY is set.
    Use arXiv, PubMed, or Crossref instead (all free).
    """
    
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        self.base_url = "https://serpapi.com/search"
        
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search IEEE Xplore - only if SerpAPI key is available"""
        if not self.api_key:
            log.debug("IEEE Xplore requires SerpAPI (paid). Use arXiv, PubMed, or Crossref instead (all free).")
            return []
        
        try:
            params = {
                "engine": "google",
                "q": f"{query} site:ieee.org",
                "api_key": self.api_key,
                "num": min(max_results, 20)
            }
            
            http_client = await get_http_client()
            response = await http_client.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results[:max_results]:
                # Filter for IEEE Xplore links
                link = result.get("link", "")
                if "ieee.org" in link and ("xplore" in link or "ieee" in link.lower()):
                    normalized = self.normalize_paper(result)
                    if normalized:
                        papers.append(normalized)
            
            log.info(f"Found {len(papers)} papers from IEEE Xplore for '{query}'")
            return papers
            
        except Exception as e:
            log.error(f"Error searching IEEE Xplore: {e}")
            return []
        
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize IEEE Xplore result"""
        try:
            title = paper_data.get("title", "")
            link = paper_data.get("link", "")
            snippet = paper_data.get("snippet", "")
            
            if not title or not link:
                return None
            
            return {
                "title": title,
                "url": link,
                "abstract": snippet,
                "source": "IEEE Xplore",
                "type": "paper"
            }
        except Exception as e:
            log.warning(f"Error normalizing IEEE paper: {e}")
            return None


class PubMedAdapter(AcademicAPIAdapter):
    """
    PubMed adapter for medical/health sciences papers
    
    PubMed provides a free API (E-utilities) for searching.
    Rate limit: 3 requests/second without API key, 10/second with key
    """
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = os.getenv("NCBI_API_KEY")  # Optional but recommended
        self.rate_limit_delay = 0.34 if self.api_key else 0.34  # 3 requests/second = 0.34s delay
        
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search PubMed for papers with retry logic and rate limiting"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Step 1: Search and get IDs
                search_params = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": min(max_results, 100),
                    "retmode": "json"
                }
                
                if self.api_key:
                    search_params["api_key"] = self.api_key
                
                http_client = await get_http_client()
                search_url = f"{self.base_url}/esearch.fcgi"
                
                # Add rate limiting delay
                await asyncio.sleep(self.rate_limit_delay)
                
                search_response = await http_client.get(search_url, params=search_params, timeout=15)
                search_response.raise_for_status()
                search_data = search_response.json()
                
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                if not pmids:
                    return []
                
                # Step 2: Fetch details for each paper
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(pmids[:max_results]),
                    "retmode": "xml"
                }
                
                if self.api_key:
                    fetch_params["api_key"] = self.api_key
                
                fetch_url = f"{self.base_url}/efetch.fcgi"
                
                # Add rate limiting delay
                await asyncio.sleep(self.rate_limit_delay)
                
                fetch_response = await http_client.get(fetch_url, params=fetch_params, timeout=15)
                fetch_response.raise_for_status()
                
                # Parse XML in executor to avoid blocking event loop
                import xml.etree.ElementTree as ET
                root = ET.fromstring(fetch_response.text)
                
                papers = []
                for article in root.findall('.//PubmedArticle'):
                    normalized = self._parse_pubmed_article(article)
                    if normalized:
                        papers.append(normalized)
                
                log.info(f"Found {len(papers)} papers from PubMed for '{query}'")
                return papers
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        log.warning(f"PubMed rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        log.error(f"PubMed rate limit exceeded after {max_retries} attempts")
                        return []
                else:
                    log.error(f"PubMed HTTP error: {e}")
                    return []
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    log.warning(f"PubMed error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    log.error(f"Error searching PubMed after {max_retries} attempts: {e}")
                    return []
        
        return []
    
    def _parse_pubmed_article(self, article) -> Optional[Dict[str, Any]]:
        """Parse a single PubMed article"""
        try:
            # Title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_elems = article.findall('.//AbstractText')
            abstract = " ".join([elem.text for elem in abstract_elems if elem.text])
            
            # Authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Year
            year = None
            pub_date = article.find('.//PubDate/Year')
            if pub_date is not None:
                try:
                    year = int(pub_date.text)
                except (ValueError, TypeError, AttributeError):
                    pass
            
            # URL (PubMed link)
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ""
            
            if not title:
                return None
            
            return {
                "title": title,
                "url": url,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "source": "PubMed",
                "type": "paper"
            }
        except Exception as e:
            log.warning(f"Error parsing PubMed article: {e}")
            return None
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize PubMed paper (already normalized in _parse_pubmed_article)"""
        return paper_data


class GoogleBooksAdapter(AcademicAPIAdapter):
    """
    Google Books API adapter
    
    Provides access to millions of books with metadata, previews, and full texts.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_BOOKS_API_KEY")  # Optional, but recommended
        self.base_url = "https://www.googleapis.com/books/v1/volumes"
        
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Google Books doesn't have papers, return empty"""
        return []
    
    async def search_books(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google Books for books"""
        try:
            params = {
                "q": query,
                "maxResults": min(max_results, 40),  # Google Books limit
                "orderBy": "relevance"
            }
            
            if self.api_key:
                params["key"] = self.api_key
            
            http_client = await get_http_client()
            response = await http_client.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            books = []
            items = data.get("items", [])
            
            for item in items[:max_results]:
                normalized = self.normalize_book(item)
                if normalized:
                    books.append(normalized)
            
            log.info(f"Found {len(books)} books from Google Books for '{query}'")
            return books
            
        except Exception as e:
            log.error(f"Error searching Google Books: {e}")
            return []
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Not applicable for Google Books"""
        return {}
    
    def normalize_book(self, book_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Google Books result to standard format"""
        try:
            volume_info = book_data.get("volumeInfo", {})
            title = volume_info.get("title", "")
            
            if not title:
                return None
            
            # Extract authors
            authors = volume_info.get("authors", [])
            
            # Extract description
            description = volume_info.get("description", "")
            if len(description) > 500:
                description = description[:500] + "..."
            
            # Extract published date
            published_date = volume_info.get("publishedDate", "")
            year = None
            if published_date:
                try:
                    year = int(published_date[:4]) if len(published_date) >= 4 else None
                except (ValueError, TypeError):
                    pass
            
            # Extract ISBN
            isbn = None
            industry_identifiers = volume_info.get("industryIdentifiers", [])
            for identifier in industry_identifiers:
                if identifier.get("type") == "ISBN_13":
                    isbn = identifier.get("identifier")
                    break
                elif identifier.get("type") == "ISBN_10" and not isbn:
                    isbn = identifier.get("identifier")
            
            # Extract categories/subjects
            categories = volume_info.get("categories", [])
            
            # Extract page count
            page_count = volume_info.get("pageCount")
            
            # Get preview link or info link
            preview_link = volume_info.get("previewLink", "")
            info_link = volume_info.get("infoLink", "")
            url = preview_link or info_link or f"https://books.google.com/books?id={book_data.get('id', '')}"
            
            # Extract publisher
            publisher = volume_info.get("publisher", "")
            
            return {
                "title": title,
                "url": url,
                "authors": authors,
                "year": year,
                "description": description,
                "isbn": isbn,
                "categories": categories,
                "page_count": page_count,
                "publisher": publisher,
                "source": "Google Books",
                "type": "book"
            }
        except Exception as e:
            log.warning(f"Error normalizing Google Books result: {e}")
            return None


class OpenLibraryAdapter(AcademicAPIAdapter):
    """
    Open Library API adapter
    
    Provides access to millions of book records with bibliographic information.
    """
    
    def __init__(self):
        self.base_url = "https://openlibrary.org"
        
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Open Library doesn't have papers, return empty"""
        return []
    
    async def search_books(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Open Library for books"""
        try:
            params = {
                "q": query,
                "limit": min(max_results, 100)  # Open Library limit
            }
            
            http_client = await get_http_client()
            search_url = f"{self.base_url}/search.json"
            response = await http_client.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            books = []
            docs = data.get("docs", [])
            
            for doc in docs[:max_results]:
                normalized = self.normalize_book(doc)
                if normalized:
                    books.append(normalized)
            
            log.info(f"Found {len(books)} books from Open Library for '{query}'")
            return books
            
        except Exception as e:
            log.error(f"Error searching Open Library: {e}")
            return []
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Not applicable for Open Library"""
        return {}
    
    def normalize_book(self, book_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Open Library result to standard format"""
        try:
            title = book_data.get("title", "")
            
            if not title:
                return None
            
            # Extract authors
            authors = []
            author_names = book_data.get("author_name", [])
            if isinstance(author_names, list):
                authors = author_names
            elif isinstance(author_names, str):
                authors = [author_names]
            
            # Extract first publish year
            first_publish_year = book_data.get("first_publish_year")
            year = int(first_publish_year) if first_publish_year else None
            
            # Extract ISBN
            isbn = None
            isbn_list = book_data.get("isbn", [])
            if isbn_list:
                isbn = isbn_list[0]  # Take first ISBN
            
            # Extract subjects
            subjects = book_data.get("subject", [])
            if isinstance(subjects, str):
                subjects = [subjects]
            
            # Extract number of pages
            number_of_pages = book_data.get("number_of_pages_median")
            
            # Build URL
            key = book_data.get("key", "")
            url = f"{self.base_url}{key}" if key else ""
            
            # Extract publisher
            publisher = book_data.get("publisher", [])
            if isinstance(publisher, list) and publisher:
                publisher = publisher[0]
            elif not isinstance(publisher, str):
                publisher = ""
            
            return {
                "title": title,
                "url": url,
                "authors": authors,
                "year": year,
                "isbn": isbn,
                "categories": subjects[:5] if subjects else [],  # Limit categories
                "page_count": number_of_pages,
                "publisher": publisher,
                "source": "Open Library",
                "type": "book"
            }
        except Exception as e:
            log.warning(f"Error normalizing Open Library result: {e}")
            return None


class CrossrefAdapter(AcademicAPIAdapter):
    """
    Crossref API adapter
    
    Provides access to scholarly works including papers, books, and other publications.
    """
    
    def __init__(self):
        self.base_url = "https://api.crossref.org"
        self.user_agent = os.getenv("CROSSREF_USER_AGENT", "YourApp/1.0 (mailto:your@email.com)")
        
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Crossref for scholarly works"""
        try:
            params = {
                "query": query,
                "rows": min(max_results, 100),  # Crossref limit
                "sort": "relevance"
            }
            
            headers = {
                "User-Agent": self.user_agent
            }
            
            http_client = await get_http_client()
            search_url = f"{self.base_url}/works"
            response = await http_client.get(search_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            items = data.get("message", {}).get("items", [])
            
            for item in items[:max_results]:
                normalized = self.normalize_paper(item)
                if normalized:
                    papers.append(normalized)
            
            log.info(f"Found {len(papers)} works from Crossref for '{query}'")
            return papers
            
        except Exception as e:
            log.error(f"Error searching Crossref: {e}")
            return []
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Crossref result to standard format"""
        try:
            title_list = paper_data.get("title", [])
            title = title_list[0] if title_list else ""
            
            if not title:
                return None
            
            # Extract authors
            authors = []
            author_list = paper_data.get("author", [])
            for author in author_list:
                given = author.get("given", "")
                family = author.get("family", "")
                if given or family:
                    authors.append(f"{given} {family}".strip())
            
            # Extract published date
            published_date = paper_data.get("published-print", {}) or paper_data.get("published-online", {})
            year = None
            if published_date:
                date_parts = published_date.get("date-parts", [])
                if date_parts and date_parts[0]:
                    year = int(date_parts[0][0]) if len(date_parts[0]) > 0 else None
            
            # Extract abstract
            abstract = ""
            if "abstract" in paper_data:
                abstract = paper_data["abstract"]
            
            # Extract DOI and build URL
            doi = paper_data.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else ""
            
            # Extract type
            work_type = paper_data.get("type", "article")
            
            # Extract journal/publisher
            container_title = paper_data.get("container-title", [])
            publisher = container_title[0] if container_title else ""
            
            # Extract subjects
            subjects = paper_data.get("subject", [])
            
            # Determine material type
            material_type = "paper" if work_type in ["article", "paper", "proceeding"] else "book"
            
            return {
                "title": title,
                "url": url,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "doi": doi,
                "work_type": work_type,
                "publisher": publisher,
                "categories": subjects,
                "source": "Crossref",
                "type": material_type
            }
        except Exception as e:
            log.warning(f"Error normalizing Crossref result: {e}")
            return None


class WikibooksAdapter(AcademicAPIAdapter):
    """
    Wikibooks API adapter
    
    Provides access to open-content textbooks and educational materials.
    """
    
    def __init__(self):
        self.base_url = "https://en.wikibooks.org/w/api.php"
        
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Wikibooks for educational content"""
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(max_results, 50),  # Wikibooks limit
                "format": "json"
            }
            
            http_client = await get_http_client()
            response = await http_client.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            books = []
            search_results = data.get("query", {}).get("search", [])
            
            for result in search_results[:max_results]:
                normalized = self.normalize_book(result)
                if normalized:
                    books.append(normalized)
            
            log.info(f"Found {len(books)} books from Wikibooks for '{query}'")
            return books
            
        except Exception as e:
            log.error(f"Error searching Wikibooks: {e}")
            return []
    
    def normalize_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Not applicable for Wikibooks"""
        return {}
    
    def normalize_book(self, book_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Wikibooks result to standard format"""
        try:
            title = book_data.get("title", "")
            snippet = book_data.get("snippet", "")
            
            if not title:
                return None
            
            # Build URL
            page_title = title.replace(" ", "_")
            url = f"https://en.wikibooks.org/wiki/{page_title}"
            
            # Extract word count as page estimate
            word_count = book_data.get("size", 0)
            estimated_pages = word_count // 250 if word_count > 0 else None  # Rough estimate
            
            return {
                "title": title,
                "url": url,
                "description": snippet,
                "page_count": estimated_pages,
                "source": "Wikibooks",
                "type": "book"
            }
        except Exception as e:
            log.warning(f"Error normalizing Wikibooks result: {e}")
            return None


class AcademicAPIManager:
    """Manages all academic API adapters (free APIs only)"""
    
    def __init__(self):
        # Only include free APIs (no SerpAPI required)
        self.adapters = {
            "arxiv": ArxivAdapter(),
            "pubmed": PubMedAdapter(),
            "google_books": GoogleBooksAdapter(),
            "open_library": OpenLibraryAdapter(),
            "crossref": CrossrefAdapter(),
            "wikibooks": WikibooksAdapter()
        }
        
        # Optional: Include paid APIs only if SerpAPI key is available
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            log.info("SerpAPI key found - enabling Google Scholar and IEEE Xplore")
            self.adapters["google_scholar"] = GoogleScholarAdapter()
            self.adapters["ieee"] = IEEEXploreAdapter()
        else:
            log.info("SerpAPI key not found - using free APIs only (arXiv, PubMed, Crossref, Google Books, Open Library, Wikibooks)")
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapter names"""
        return list(self.adapters.keys())
    
    async def search_all_sources(
        self, 
        query: str, 
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 5,
        material_type: str = "paper"
    ) -> List[Dict[str, Any]]:
        """
        Search all academic sources for papers or books
        
        Args:
            query: Search query
            sources: List of source names to search (None = all)
            max_results_per_source: Max results per source
            material_type: "paper" or "book"
            
        Returns:
            List of normalized materials from all sources
        """
        if sources is None:
            sources = list(self.adapters.keys())
        
        all_materials = []
        
        for source_name in sources:
            if source_name not in self.adapters:
                log.warning(f"Unknown academic source: {source_name}")
                continue
            
            try:
                adapter = self.adapters[source_name]
                
                if material_type == "book":
                    # Search for books
                    materials = await adapter.search_books(query, max_results=max_results_per_source)
                else:
                    # Search for papers
                    materials = await adapter.search_papers(query, max_results=max_results_per_source)
                
                all_materials.extend(materials)
            except Exception as e:
                log.error(f"Error searching {source_name}: {e}")
                continue
        
        log.info(f"Found {len(all_materials)} total {material_type}s from {len(sources)} sources")
        return all_materials
    
    async def search_books(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 5
    ) -> List[Dict[str, Any]]:
        """Convenience method to search for books"""
        book_sources = ["google_books", "open_library", "wikibooks", "crossref"]
        if sources is None:
            sources = book_sources
        else:
            sources = [s for s in sources if s in book_sources]
        
        return await self.search_all_sources(query, sources, max_results_per_source, material_type="book")

