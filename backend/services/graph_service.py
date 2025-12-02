"""
Graph Service
Handles PageRank, CheiRank computation and call graph analysis
"""

import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(self):
        self.damping_factor = 0.85
        self.max_iterations = 100
        self.tolerance = 1e-6
    
    async def build_and_analyze(
        self,
        owner: str,
        name: str,
        file_pattern: str = "*.c",
        compute_pagerank: bool = True,
        compute_cheirank: bool = True
    ) -> Dict[str, Any]:
        """
        Build call graph from repository and compute rankings
        
        Args:
            owner: Repository owner
            name: Repository name
            file_pattern: File pattern to analyze
            compute_pagerank: Whether to compute PageRank
            compute_cheirank: Whether to compute CheiRank
        """
        try:
            # For MVP, we'll create a simplified graph
            # In production, you'd parse actual C code to extract function calls
            
            # Placeholder: Create sample graph for demonstration
            graph = self._create_sample_kernel_graph()
            
            results = {
                "repository": f"{owner}/{name}",
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "graph_metrics": {
                    "density": nx.density(graph),
                    "is_strongly_connected": nx.is_strongly_connected(graph) if graph.is_directed() else None
                }
            }
            
            if compute_pagerank:
                pagerank = self.compute_pagerank_from_graph(graph)
                results["pagerank"] = pagerank
                results["top_pagerank_nodes"] = self._get_top_nodes(pagerank, 10)
            
            if compute_cheirank:
                cheirank = self.compute_cheirank_from_graph(graph)
                results["cheirank"] = cheirank
                results["top_cheirank_nodes"] = self._get_top_nodes(cheirank, 10)
            
            # Identify critical nodes (high in both rankings)
            if compute_pagerank and compute_cheirank:
                results["critical_nodes"] = self._identify_critical_nodes(
                    pagerank, cheirank, top_n=10
                )
            
            return results
        
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            return {"error": str(e)}
    
    def compute_pagerank(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Compute PageRank for a given graph
        
        Args:
            nodes: List of node identifiers
            edges: List of directed edges (from, to)
        """
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        return self.compute_pagerank_from_graph(G)
    
    def compute_pagerank_from_graph(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute PageRank from NetworkX graph"""
        try:
            pagerank = nx.pagerank(
                graph,
                alpha=self.damping_factor,
                max_iter=self.max_iterations,
                tol=self.tolerance
            )
            return {str(k): float(v) for k, v in pagerank.items()}
        except Exception as e:
            logger.error(f"Error computing PageRank: {str(e)}")
            return {}
    
    def compute_cheirank(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Compute CheiRank (inverse PageRank) for a given graph
        
        CheiRank ranks nodes by their outgoing influence
        """
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        return self.compute_cheirank_from_graph(G)
    
    def compute_cheirank_from_graph(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute CheiRank from NetworkX graph"""
        try:
            # CheiRank is PageRank on the reversed graph
            reversed_graph = graph.reverse()
            cheirank = nx.pagerank(
                reversed_graph,
                alpha=self.damping_factor,
                max_iter=self.max_iterations,
                tol=self.tolerance
            )
            return {str(k): float(v) for k, v in cheirank.items()}
        except Exception as e:
            logger.error(f"Error computing CheiRank: {str(e)}")
            return {}
    
    async def build_citation_graph(
        self,
        papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build citation graph for academic papers
        
        Args:
            papers: List of paper metadata from PubMed or similar
        """
        try:
            G = nx.DiGraph()
            
            # Add papers as nodes
            for paper in papers:
                pmid = paper.get("pmid", paper.get("id"))
                G.add_node(pmid, **paper)
            
            # In production, you'd fetch actual citation data
            # For now, create sample connections based on MeSH terms
            for i, paper1 in enumerate(papers):
                pmid1 = paper1.get("pmid", paper1.get("id"))
                mesh1 = set(paper1.get("mesh_terms", []))
                
                for paper2 in papers[i+1:]:
                    pmid2 = paper2.get("pmid", paper2.get("id"))
                    mesh2 = set(paper2.get("mesh_terms", []))
                    
                    # Create edge if papers share MeSH terms (similarity)
                    overlap = len(mesh1 & mesh2)
                    if overlap > 0:
                        G.add_edge(pmid1, pmid2, weight=overlap)
            
            # Compute rankings
            pagerank = self.compute_pagerank_from_graph(G) if G.number_of_nodes() > 0 else {}
            cheirank = self.compute_cheirank_from_graph(G) if G.number_of_nodes() > 0 else {}
            
            return {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "pagerank": pagerank,
                "cheirank": cheirank,
                "top_papers": self._get_top_nodes(pagerank, 5)
            }
        
        except Exception as e:
            logger.error(f"Error building citation graph: {str(e)}")
            return {"error": str(e)}
    
    def _create_sample_kernel_graph(self) -> nx.DiGraph:
        """Create a sample Linux kernel call graph for demonstration"""
        G = nx.DiGraph()
        
        # Sample functions from Linux networking stack
        functions = [
            "sys_socket", "sys_bind", "sys_connect", "sys_send", "sys_recv",
            "tcp_v4_connect", "tcp_sendmsg", "tcp_recvmsg", "tcp_v4_rcv",
            "ip_queue_xmit", "ip_rcv", "ip_local_deliver",
            "__netif_receive_skb", "dev_queue_xmit", "netif_rx",
            "sk_alloc", "sk_free", "__sk_dst_check",
            "tcp_v4_init_sock", "tcp_connect", "tcp_transmit_skb"
        ]
        
        G.add_nodes_from(functions)
        
        # Sample call relationships (caller -> callee)
        edges = [
            ("sys_socket", "sk_alloc"),
            ("sys_connect", "tcp_v4_connect"),
            ("tcp_v4_connect", "tcp_connect"),
            ("tcp_connect", "tcp_transmit_skb"),
            ("sys_send", "tcp_sendmsg"),
            ("tcp_sendmsg", "tcp_transmit_skb"),
            ("tcp_transmit_skb", "ip_queue_xmit"),
            ("ip_queue_xmit", "dev_queue_xmit"),
            ("sys_recv", "tcp_recvmsg"),
            ("netif_rx", "__netif_receive_skb"),
            ("__netif_receive_skb", "ip_rcv"),
            ("ip_rcv", "ip_local_deliver"),
            ("ip_local_deliver", "tcp_v4_rcv"),
            ("tcp_v4_rcv", "tcp_recvmsg"),
            ("tcp_v4_init_sock", "sk_alloc"),
            ("sk_free", "__sk_dst_check")
        ]
        
        G.add_edges_from(edges)
        
        return G
    
    def _get_top_nodes(
        self,
        rankings: Dict[str, float],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top N nodes by ranking score"""
        sorted_nodes = sorted(
            rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"node": node, "score": score}
            for node, score in sorted_nodes[:top_n]
        ]
    
    def _identify_critical_nodes(
        self,
        pagerank: Dict[str, float],
        cheirank: Dict[str, float],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify critical nodes that rank high in both PageRank and CheiRank
        
        These nodes are both heavily depended upon (PageRank) and
        have wide influence (CheiRank) - prime targets for security focus
        """
        # Combine scores (geometric mean)
        combined_scores = {}
        all_nodes = set(pagerank.keys()) | set(cheirank.keys())
        
        for node in all_nodes:
            pr = pagerank.get(node, 0)
            cr = cheirank.get(node, 0)
            # Geometric mean
            combined_scores[node] = (pr * cr) ** 0.5
        
        # Sort by combined score
        sorted_nodes = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                "node": node,
                "combined_score": score,
                "pagerank": pagerank.get(node, 0),
                "cheirank": cheirank.get(node, 0),
                "criticality": "HIGH" if score > 0.01 else "MEDIUM" if score > 0.005 else "LOW"
            }
            for node, score in sorted_nodes[:top_n]
        ]
