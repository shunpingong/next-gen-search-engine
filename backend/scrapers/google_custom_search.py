"""
Google Custom Search API scraper implementation.
"""
import requests
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from .base_scraper import BaseScraper
from .config import Config
from .utils import enrich_result_with_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCustomSearchScraper(BaseScraper):
    """Scraper using Google Custom Search JSON API."""
    
    def __init__(self):
        super().__init__('google_custom_search')
        self.api_key = Config.GOOGLE_API_KEY
        self.search_engine_id = Config.GOOGLE_SEARCH_ENGINE_ID
        self.base_url = 'https://www.googleapis.com/customsearch/v1'
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search using Google Custom Search API.
        
        Args:
            query: Search query string
            top_k: Number of results to return (max 10 per request)
            
        Returns:
            List of search results with enriched metadata
        """
        if not self.api_key or not self.search_engine_id:
            logger.error("Google API key or Search Engine ID not configured")
            return []
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(top_k, 10)  # Google limits to 10 per request
            }
            
            response = requests.get(self.base_url, params=params, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Store raw data
            self.raw_data = data
            
            items = data.get('items', [])
            
            # Extract basic information
            results = []
            for item in items:
                result = {
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'description': item.get('snippet', ''),
                }
                
                # Add pagemap data if available
                pagemap = item.get('pagemap', {})
                if 'metatags' in pagemap and pagemap['metatags']:
                    meta = pagemap['metatags'][0]
                    result['meta_description'] = meta.get('og:description', meta.get('description', ''))
                    result['meta_title'] = meta.get('og:title', meta.get('title', ''))
                
                results.append(result)
            
            # Enrich results with full page content
            logger.info(f"Fetching full content for {len(results)} results...")
            with ThreadPoolExecutor(max_workers=Config.MAX_THREADS) as executor:
                enriched_results = list(executor.map(
                    lambda r: enrich_result_with_metadata(r, Config.REQUEST_TIMEOUT),
                    results
                ))
            
            self.results = enriched_results
            logger.info(f"Google Custom Search completed: {len(self.results)} results")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in Google Custom Search: {e}")
            return []


if __name__ == "__main__":
    
    # Example usage: run this file directly
    scraper = GoogleCustomSearchScraper()

    if not scraper.api_key or not scraper.search_engine_id:
        print("Please set your Google API key and Search Engine ID in Config.GOOGLE_API_KEY and Config.GOOGLE_SEARCH_ENGINE_ID")
    else:
        query = "Best places to visit in Japan"
        top_k = 5
        output_dir = Config.OUTPUT_DIR

        # Run the search
        results = scraper.search(query, top_k=top_k)

        if results:
            # Save both raw and cleaned results
            filepaths = scraper.save_both(query, output_dir)
            print(f"Raw data saved to: {filepaths['raw_file']}")
            print(f"Cleaned results saved to: {filepaths['cleaned_file']}")

            # Print summary
            summary = scraper.get_metadata_summary()
            print("Summary:", summary)

            # Optional: print top results
            print(f"Top {top_k} results for query: '{query}'")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} - {result['url']}")
