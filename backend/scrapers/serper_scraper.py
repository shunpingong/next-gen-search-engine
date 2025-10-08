"""
Serper API scraper implementation.
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


class SerperScraper(BaseScraper):
    """Scraper using Serper API."""
    
    def __init__(self):
        super().__init__('serper')
        self.api_key = Config.SERPER_API_KEY
        self.base_url = 'https://google.serper.dev/search'
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search using Serper API.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with enriched metadata
        """
        if not self.api_key:
            logger.error("Serper API key not configured")
            return []
        
        try:
            logger.info(f"Searching Serper for: {query}")
            
            payload = {
                'q': query,
                'num': top_k
            }
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            # Store raw data
            self.raw_data = data
            
            results = []
            for item in data.get('organic', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'description': item.get('snippet', ''),
                    'snippet': item.get('snippet', ''),
                    'position': item.get('position', 0),
                    'date': item.get('date', '')
                })
            
            # Enrich results with full page content
            logger.info(f"Fetching full content for {len(results)} results...")
            with ThreadPoolExecutor(max_workers=Config.MAX_THREADS) as executor:
                enriched_results = list(executor.map(
                    lambda r: enrich_result_with_metadata(r, Config.REQUEST_TIMEOUT),
                    results
                ))
            
            self.results = enriched_results
            logger.info(f"Serper search completed: {len(self.results)} results")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in Serper search: {e}")
            return []


if __name__ == "__main__":
    
    # Example usage: run this file directly
    scraper = SerperScraper()

    if not scraper.api_key:
        print("Please set your Serper API key in Config.SERPER_API_KEY")
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
