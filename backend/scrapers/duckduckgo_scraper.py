"""
DuckDuckGo search scraper using ddgs library.
"""
from duckduckgo_search import DDGS
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from .base_scraper import BaseScraper
from .config import Config
from .utils import enrich_result_with_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuckDuckGoScraper(BaseScraper):
    """Scraper using ddgs library (free, no API key needed)."""
    
    def __init__(self):
        super().__init__('duckduckgo')
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search using DuckDuckGo.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with enriched metadata
        """
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            # Use DDGS with improved error handling
            results = []
            raw_search_results = []
            try:
                ddgs = DDGS()
                search_results = ddgs.text(query, max_results=top_k)
                
                for result in search_results:
                    raw_search_results.append(result)  # Store raw result
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'description': result.get('body', ''),
                        'snippet': result.get('body', '')
                    })
                
                # Store raw data
                self.raw_data = {
                    'query': query,
                    'max_results': top_k,
                    'results': raw_search_results
                }
                    
            except Exception as search_error:
                logger.error(f"DuckDuckGo search API error: {search_error}")
                return []
            
            if not results:
                logger.warning(f"No results found for query: {query}")
                return []
            
            logger.info(f"Found {len(results)} initial results from DuckDuckGo")
            
            # Enrich results with full page content (with better error handling)
            logger.info(f"Fetching full content for {len(results)} results...")
            enriched_results = []
            
            with ThreadPoolExecutor(max_workers=min(Config.MAX_THREADS, len(results))) as executor:
                # Submit all tasks
                future_to_result = {
                    executor.submit(enrich_result_with_metadata, result, Config.REQUEST_TIMEOUT): result 
                    for result in results
                }
                
                # Collect results as they complete
                for future in future_to_result:
                    try:
                        enriched_result = future.result()
                        enriched_results.append(enriched_result)
                    except Exception as e:
                        # If enrichment fails, use original result
                        original_result = future_to_result[future]
                        logger.warning(f"Failed to enrich result for {original_result.get('url', 'unknown URL')}: {e}")
                        enriched_results.append(original_result)
            
            self.results = enriched_results
            logger.info(f"DuckDuckGo search completed: {len(self.results)} results ({len([r for r in self.results if 'content' in r and r['content']])} enriched)")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return []


if __name__ == "__main__":
    
    # Example usage: run this file directly
    scraper = DuckDuckGoScraper()

    # DuckDuckGo doesn't require API key
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
