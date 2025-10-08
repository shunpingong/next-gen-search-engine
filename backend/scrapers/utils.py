"""
Utility functions for scraping and metadata extraction.
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_page_content(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch the HTML content of a page with retry logic and better headers.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        HTML content as string or None if failed
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # Try up to 3 times with different strategies
    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(headers)
            
            # Add slight delay for subsequent attempts
            if attempt > 0:
                import time
                time.sleep(1)
            
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.RequestException as e:
            if attempt == 2:  # Last attempt
                logger.error(f"Error fetching {url} after {attempt + 1} attempts: {e}")
                return None
            else:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    return None


def extract_metadata(html: str, url: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML content.
    
    Args:
        html: HTML content as string
        url: Original URL
        
    Returns:
        Dictionary containing extracted metadata
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    metadata = {
        'url': url,
        'title': '',
        'description': '',
        'keywords': '',
        'content': '',
        'headings': [],
        'links_count': 0,
        'images_count': 0,
    }
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text().strip()
    
    # Extract meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        metadata['description'] = meta_desc.get('content').strip()
    
    # Extract keywords
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords and meta_keywords.get('content'):
        metadata['keywords'] = meta_keywords.get('content').strip()
    
    # Extract Open Graph data
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        metadata['og_title'] = og_title.get('content').strip()
    
    og_desc = soup.find('meta', property='og:description')
    if og_desc and og_desc.get('content'):
        metadata['og_description'] = og_desc.get('content').strip()
    
    # Extract main content (paragraphs)
    paragraphs = soup.find_all('p')
    content_text = ' '.join([p.get_text().strip() for p in paragraphs[:10]])  # First 10 paragraphs
    metadata['content'] = content_text[:1000]  # Limit to 1000 chars
    
    # Extract headings
    headings = []
    for i in range(1, 7):
        for heading in soup.find_all(f'h{i}'):
            headings.append({
                'level': i,
                'text': heading.get_text().strip()
            })
    metadata['headings'] = headings[:10]  # First 10 headings
    
    # Count links and images
    metadata['links_count'] = len(soup.find_all('a'))
    metadata['images_count'] = len(soup.find_all('img'))
    
    return metadata


def enrich_result_with_metadata(result: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
    """
    Enrich a search result with additional metadata from the page.
    
    Args:
        result: Dictionary containing at least a 'url' field
        timeout: Request timeout in seconds
        
    Returns:
        Enriched result dictionary
    """
    url = result.get('url') or result.get('link')
    if not url:
        return result
    
    html = fetch_page_content(url, timeout)
    if html:
        metadata = extract_metadata(html, url)
        # Merge metadata with existing result
        enriched = {**result, **metadata}
        return enriched
    
    return result
