"""
PubMed E-utilities API Service
Handles queries to PubMed for medical literature search
"""

import aiohttp
from typing import List, Dict, Any, Optional
import logging
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class PubMedService:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        # Consider registering for an API key at https://www.ncbi.nlm.nih.gov/account/
        self.email = "your-email@example.com"  # Required by NCBI
    
    async def search(
        self,
        query: str,
        specialty: Optional[str] = None,
        max_results: int = 10,
        sort: str = "relevance"
    ) -> Dict[str, Any]:
        """
        Search PubMed literature
        
        Args:
            query: Search query
            specialty: Medical specialty filter
            max_results: Maximum number of results
            sort: Sort order (relevance, date, citations)
        """
        try:
            # Build search query
            search_term = query
            if specialty:
                search_term = f"{query} AND {specialty}[MeSH Terms]"
            
            # Step 1: Search to get PMIDs
            pmids = await self._search_pmids(search_term, max_results, sort)
            
            if not pmids:
                return {"total_count": 0, "articles": []}
            
            # Step 2: Fetch article details
            articles = await self._fetch_articles(pmids)
            
            return {
                "total_count": len(pmids),
                "query": query,
                "articles": articles
            }
        
        except Exception as e:
            logger.error(f"Error in PubMed search: {str(e)}")
            return {"total_count": 0, "articles": [], "error": str(e)}
    
    async def get_article(self, pmid: str) -> Dict[str, Any]:
        """Get detailed information for a specific article"""
        articles = await self._fetch_articles([pmid])
        return articles[0] if articles else {}
    
    async def get_related(self, pmid: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get related articles for a given PMID"""
        try:
            url = f"{self.base_url}/elink.fcgi"
            params = {
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "retmode": "json",
                "email": self.email
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract related PMIDs
                        linksets = data.get("linksets", [])
                        if linksets:
                            linksetdbs = linksets[0].get("linksetdbs", [])
                            if linksetdbs:
                                related_pmids = [
                                    str(link) for link in linksetdbs[0].get("links", [])
                                ][:max_results]
                                
                                # Fetch article details
                                return await self._fetch_articles(related_pmids)
                        
                        return []
                    else:
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching related articles: {str(e)}")
            return []
    
    async def _search_pmids(
        self,
        query: str,
        max_results: int,
        sort: str
    ) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        try:
            url = f"{self.base_url}/esearch.fcgi"
            
            sort_param = "relevance"
            if sort == "date":
                sort_param = "pub_date"
            
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": sort_param,
                "email": self.email
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        esearchresult = data.get("esearchresult", {})
                        return esearchresult.get("idlist", [])
                    else:
                        logger.error(f"PubMed search error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    async def _fetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article details for a list of PMIDs"""
        if not pmids:
            return []
        
        try:
            url = f"{self.base_url}/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "email": self.email
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        return self._parse_articles_xml(xml_data)
                    else:
                        logger.error(f"PubMed fetch error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching articles: {str(e)}")
            return []
    
    def _parse_articles_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parse XML response from PubMed"""
        articles = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article_elem in root.findall(".//PubmedArticle"):
                article = self._parse_article_element(article_elem)
                if article:
                    articles.append(article)
        
        except Exception as e:
            logger.error(f"Error parsing XML: {str(e)}")
        
        return articles
    
    def _parse_article_element(self, article_elem) -> Dict[str, Any]:
        """Parse a single article element from XML"""
        try:
            medline = article_elem.find(".//MedlineCitation")
            article_data = medline.find(".//Article")
            
            # PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Title
            title_elem = article_data.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_texts = article_data.findall(".//AbstractText")
            abstract = " ".join([
                at.text for at in abstract_texts if at.text
            ])
            
            # Authors
            authors = []
            for author_elem in article_data.findall(".//Author"):
                lastname = author_elem.find(".//LastName")
                forename = author_elem.find(".//ForeName")
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            
            # Journal
            journal_elem = article_data.find(".//Journal")
            journal = ""
            if journal_elem is not None:
                title_elem = journal_elem.find(".//Title")
                journal = title_elem.text if title_elem is not None else ""
            
            # Publication date
            pub_date_elem = article_data.find(".//PubDate")
            pub_date = ""
            if pub_date_elem is not None:
                year = pub_date_elem.find(".//Year")
                month = pub_date_elem.find(".//Month")
                day = pub_date_elem.find(".//Day")
                
                date_parts = []
                if year is not None:
                    date_parts.append(year.text)
                if month is not None:
                    date_parts.append(month.text)
                if day is not None:
                    date_parts.append(day.text)
                
                pub_date = "-".join(date_parts)
            
            # MeSH terms
            mesh_terms = []
            for mesh_elem in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "publication_date": pub_date,
                "mesh_terms": mesh_terms,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
        
        except Exception as e:
            logger.error(f"Error parsing article element: {str(e)}")
            return {}
