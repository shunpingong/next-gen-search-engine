"""
Selenium-based scraper for multiple search engines.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from typing import List, Dict, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from .base_scraper import BaseScraper
from .config import Config
from .utils import enrich_result_with_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeleniumScraper(BaseScraper):
    """Scraper using Selenium to scrape multiple search engines."""
    
    SEARCH_ENGINES = {
        'google': 'https://www.google.com',
        'bing': 'https://www.bing.com',
        'yahoo': 'https://search.yahoo.com',
        'duckduckgo': 'https://duckduckgo.com'
    }
    
    def __init__(self, engine: str = 'google'):
        super().__init__(f'selenium_{engine}')
        self.engine = engine.lower()
        if self.engine not in self.SEARCH_ENGINES:
            raise ValueError(f"Unsupported engine: {engine}")
        self.base_url = self.SEARCH_ENGINES[self.engine]
        
    def _init_driver(self) -> webdriver.Chrome:
        """Initialize Chrome WebDriver with options."""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        return driver
    
    def _scrape_google(self, driver: webdriver.Chrome, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Scrape Google search results."""
        driver.get(self.base_url)
        time.sleep(2)
        
        # Find search box and enter query
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for results
        time.sleep(3)
        
        results = []
        search_results = driver.find_elements(By.CSS_SELECTOR, 'div.g')
        
        for result in search_results[:top_k]:
            try:
                title_elem = result.find_element(By.CSS_SELECTOR, 'h3')
                link_elem = result.find_element(By.CSS_SELECTOR, 'a')
                
                title = title_elem.text
                url = link_elem.get_attribute('href')
                
                # Try to get description
                try:
                    desc_elem = result.find_element(By.CSS_SELECTOR, 'div[data-sncf="1"], div.VwiC3b')
                    description = desc_elem.text
                except:
                    description = ''
                
                if url and url.startswith('http'):
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description,
                        'snippet': description
                    })
            except Exception as e:
                logger.debug(f"Error parsing result: {e}")
                continue
        
        return results
    
    def _scrape_bing(self, driver: webdriver.Chrome, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Scrape Bing search results."""
        driver.get(self.base_url)
        time.sleep(2)
        
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        time.sleep(3)
        
        results = []
        search_results = driver.find_elements(By.CSS_SELECTOR, 'li.b_algo')
        
        for result in search_results[:top_k]:
            try:
                title_elem = result.find_element(By.CSS_SELECTOR, 'h2 a')
                title = title_elem.text
                url = title_elem.get_attribute('href')
                
                try:
                    desc_elem = result.find_element(By.CSS_SELECTOR, 'p, div.b_caption p')
                    description = desc_elem.text
                except:
                    description = ''
                
                if url and url.startswith('http'):
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description,
                        'snippet': description
                    })
            except Exception as e:
                logger.debug(f"Error parsing result: {e}")
                continue
        
        return results
    
    def _scrape_yahoo(self, driver: webdriver.Chrome, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Scrape Yahoo search results."""
        driver.get(self.base_url)
        time.sleep(2)
        
        search_box = driver.find_element(By.NAME, 'p')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        time.sleep(3)
        
        results = []
        search_results = driver.find_elements(By.CSS_SELECTOR, 'div.dd.algo')
        
        for result in search_results[:top_k]:
            try:
                title_elem = result.find_element(By.CSS_SELECTOR, 'h3 a')
                title = title_elem.text
                url = title_elem.get_attribute('href')
                
                try:
                    desc_elem = result.find_element(By.CSS_SELECTOR, 'p')
                    description = desc_elem.text
                except:
                    description = ''
                
                if url and url.startswith('http'):
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description,
                        'snippet': description
                    })
            except Exception as e:
                logger.debug(f"Error parsing result: {e}")
                continue
        
        return results
    
    def _scrape_duckduckgo(self, driver: webdriver.Chrome, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Scrape DuckDuckGo search results."""
        driver.get(self.base_url)
        time.sleep(2)
        
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        time.sleep(3)
        
        results = []
        search_results = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="result"]')
        
        for result in search_results[:top_k]:
            try:
                title_elem = result.find_element(By.CSS_SELECTOR, 'h2 a')
                title = title_elem.text
                url = title_elem.get_attribute('href')
                
                try:
                    desc_elem = result.find_element(By.CSS_SELECTOR, 'div[data-result="snippet"]')
                    description = desc_elem.text
                except:
                    description = ''
                
                if url and url.startswith('http'):
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description,
                        'snippet': description
                    })
            except Exception as e:
                logger.debug(f"Error parsing result: {e}")
                continue
        
        return results
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search using Selenium.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with enriched metadata
        """
        driver = None
        try:
            driver = self._init_driver()
            logger.info(f"Scraping {self.engine} for query: {query}")
            
            # Route to appropriate scraper
            if self.engine == 'google':
                results = self._scrape_google(driver, query, top_k)
            elif self.engine == 'bing':
                results = self._scrape_bing(driver, query, top_k)
            elif self.engine == 'yahoo':
                results = self._scrape_yahoo(driver, query, top_k)
            elif self.engine == 'duckduckgo':
                results = self._scrape_duckduckgo(driver, query, top_k)
            else:
                results = []
            
            # Store raw data (the basic scraped results before enrichment)
            self.raw_data = {
                'engine': self.engine,
                'query': query,
                'max_results': top_k,
                'results': results
            }
            
            # Enrich results with full page content
            logger.info(f"Fetching full content for {len(results)} results...")
            with ThreadPoolExecutor(max_workers=Config.MAX_THREADS) as executor:
                enriched_results = list(executor.map(
                    lambda r: enrich_result_with_metadata(r, Config.REQUEST_TIMEOUT),
                    results
                ))
            
            self.results = enriched_results
            logger.info(f"Selenium {self.engine} scraping completed: {len(self.results)} results")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in Selenium scraping ({self.engine}): {e}")
            return []
        finally:
            if driver:
                driver.quit()


if __name__ == "__main__":
    
    # Example usage: run this file directly
    scraper = SeleniumScraper()

    # Selenium doesn't require API key but needs webdriver setup
    query = "Best places to visit in Japan"
    top_k = 5
    output_dir = Config.OUTPUT_DIR

    # Run the search (defaults to Google)
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
