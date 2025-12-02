"""
NVD (National Vulnerability Database) API Service
Handles queries to NIST NVD for CVE information
"""

import aiohttp
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class NVDService:
    def __init__(self):
        self.base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.api_key = os.getenv("NVD_API_KEY")  # Optional but recommended for rate limits
        self.headers = {}
        if self.api_key:
            self.headers["apiKey"] = self.api_key
    
    async def get_cve(self, cve_id: str) -> Dict[str, Any]:
        """
        Get details of a specific CVE
        
        Args:
            cve_id: CVE identifier (e.g., CVE-2024-1234)
        """
        try:
            params = {"cveId": cve_id}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        vulnerabilities = data.get("vulnerabilities", [])
                        if vulnerabilities:
                            return self._format_cve(vulnerabilities[0])
                        return {}
                    else:
                        logger.error(f"NVD API error: {response.status}")
                        return {}
        
        except Exception as e:
            logger.error(f"Error fetching CVE {cve_id}: {str(e)}")
            return {}
    
    async def search_vulnerabilities(
        self,
        keyword: str,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search vulnerabilities by keyword
        
        Args:
            keyword: Search keyword (product name, vendor, etc.)
            max_results: Maximum number of results
        """
        try:
            params = {
                "keywordSearch": keyword,
                "resultsPerPage": min(max_results, 100)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        vulnerabilities = data.get("vulnerabilities", [])
                        return [self._format_cve(v) for v in vulnerabilities[:max_results]]
                    else:
                        logger.error(f"NVD API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error searching NVD: {str(e)}")
            return []
    
    async def search_by_cpe(self, cpe_name: str) -> List[Dict[str, Any]]:
        """
        Search vulnerabilities by CPE (Common Platform Enumeration)
        
        Args:
            cpe_name: CPE name (e.g., cpe:2.3:o:linux:linux_kernel:*)
        """
        try:
            params = {"cpeName": cpe_name}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        vulnerabilities = data.get("vulnerabilities", [])
                        return [self._format_cve(v) for v in vulnerabilities]
                    else:
                        return []
        
        except Exception as e:
            logger.error(f"Error searching by CPE: {str(e)}")
            return []
    
    def _format_cve(self, vuln: Dict) -> Dict[str, Any]:
        """Format CVE data from NVD response"""
        cve = vuln.get("cve", {})
        
        # Extract CVSS scores
        metrics = cve.get("metrics", {})
        cvss_v3 = metrics.get("cvssMetricV31", [{}])[0] if metrics.get("cvssMetricV31") else {}
        cvss_v2 = metrics.get("cvssMetricV2", [{}])[0] if metrics.get("cvssMetricV2") else {}
        
        # Extract descriptions
        descriptions = cve.get("descriptions", [])
        description = next((d["value"] for d in descriptions if d["lang"] == "en"), "")
        
        # Extract references
        references = cve.get("references", [])
        
        # Extract affected configurations
        configurations = cve.get("configurations", [])
        
        return {
            "id": cve.get("id", ""),
            "source_identifier": cve.get("sourceIdentifier", ""),
            "published": cve.get("published", ""),
            "last_modified": cve.get("lastModified", ""),
            "vuln_status": cve.get("vulnStatus", ""),
            "description": description,
            "cvss_v3": {
                "score": cvss_v3.get("cvssData", {}).get("baseScore"),
                "severity": cvss_v3.get("cvssData", {}).get("baseSeverity"),
                "vector": cvss_v3.get("cvssData", {}).get("vectorString")
            } if cvss_v3 else None,
            "cvss_v2": {
                "score": cvss_v2.get("cvssData", {}).get("baseScore"),
                "severity": cvss_v2.get("baseSeverity"),
                "vector": cvss_v2.get("cvssData", {}).get("vectorString")
            } if cvss_v2 else None,
            "references": [
                {
                    "url": ref.get("url"),
                    "source": ref.get("source"),
                    "tags": ref.get("tags", [])
                }
                for ref in references
            ],
            "weaknesses": [
                w.get("description", [{}])[0].get("value")
                for w in cve.get("weaknesses", [])
            ],
            "configurations": configurations
        }
