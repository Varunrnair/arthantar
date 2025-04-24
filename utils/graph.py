import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import json

class KnowledgeGraphAnalyzer:
    def __init__(self, graph: Dict):
        """Initialize the analyzer with a knowledge graph"""
        self.graph = graph
        self.G = self._create_networkx_graph()
        
    def _create_networkx_graph(self) -> nx.DiGraph:
        """Convert the graph dictionary to a NetworkX graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.graph['nodes']:
            G.add_node(
                node['id'],
                type=node.get('type', 'Entity'),
                gender=node.get('gender', 'unknown')
            )
            
        # Add edges
        for rel in self.graph['relationships']:
            G.add_edge(
                rel['source'],
                rel['target'],
                type=rel['type']
            )
            
        return G
        
    def get_graph_metrics(self) -> Dict:
        """Calculate basic metrics for the graph"""
        metrics = {
            'node_count': len(self.G.nodes()),
            'edge_count': len(self.G.edges()),
            'density': nx.density(self.G),
            'is_connected': nx.is_weakly_connected(self.G),
        }
        
        # Calculate diameter if graph is connected
        if metrics['is_connected']:
            try:
                metrics['diameter'] = nx.diameter(self.G)
            except:
                metrics['diameter'] = None
                
        # Calculate average clustering coefficient
        try:
            metrics['avg_clustering'] = nx.average_clustering(self.G)
        except:
            metrics['avg_clustering'] = None
            
        return metrics
        
    def get_centrality_measures(self) -> pd.DataFrame:
        """Calculate centrality measures for all nodes"""
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(self.G)
        in_degree_centrality = nx.in_degree_centrality(self.G)
        out_degree_centrality = nx.out_degree_centrality(self.G)
        
        try:
            betweenness_centrality = nx.betweenness_centrality(self.G)
        except:
            betweenness_centrality = {node: 0 for node in self.G.nodes()}
            
        try:
            closeness_centrality = nx.closeness_centrality(self.G)
        except:
            closeness_centrality = {node: 0 for node in self.G.nodes()}
            
        # Create DataFrame
        df = pd.DataFrame({
            'Node': list(self.G.nodes()),
            'Type': [self.G.nodes[node]['type'] for node in self.G.nodes()],
            'Gender': [self.G.nodes[node]['gender'] for node in self.G.nodes()],
            'Degree': [degree_centrality[node] for node in self.G.nodes()],
            'In-Degree': [in_degree_centrality[node] for node in self.G.nodes()],
            'Out-Degree': [out_degree_centrality[node] for node in self.G.nodes()],
            'Betweenness': [betweenness_centrality[node] for node in self.G.nodes()],
            'Closeness': [closeness_centrality[node] for node in self.G.nodes()]
        })
        
        # Calculate total centrality (simple sum)
        df['Total Centrality'] = df['Degree'] + df['Betweenness'] + df['Closeness']
        
        # Sort by total centrality
        df = df.sort_values('Total Centrality', ascending=False)
        
        return df
        
    def find_communities(self) -> Dict[int, List[str]]:
        """Find communities in the graph using the Louvain method"""
        try:
            import community as community_louvain
            
            # Convert to undirected graph for community detection
            G_undirected = self.G.to_undirected()
            
            # Apply Louvain method
            partition = community_louvain.best_partition(G_undirected)
            
            # Group nodes by community
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
                
            return communities
            
        except ImportError:
            print("python-louvain package not installed. Using connected components instead.")
            # Fallback to connected components
            components = list(nx.weakly_connected_components(self.G))
            return {i: list(component) for i, component in enumerate(components)}
            
    def get_shortest_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all shortest paths between two nodes"""
        if source not in self.G or target not in self.G:
            return []
            
        try:
            paths = list(nx.all_shortest_paths(self.G, source, target))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
            
    def get_node_neighbors(self, node: str, include_types: bool = True) -> Dict:
        """Get incoming and outgoing neighbors for a node"""
        if node not in self.G:
            return {'in': [], 'out': []}
            
        in_neighbors = list(self.G.predecessors(node))
        out_neighbors = list(self.G.successors(node))
        
        if include_types:
            in_edges = [(u, self.G.edges[u, node].get('type', 'RELATED_TO')) 
                       for u in in_neighbors]
            out_edges = [(v, self.G.edges[node, v].get('type', 'RELATED_TO')) 
                        for v in out_neighbors]
            return {
                'in': in_edges,
                'out': out_edges
            }
        else:
            return {
                'in': in_neighbors,
                'out': out_neighbors
            }
            
    def export_to_json(self) -> str:
        """Export the graph to JSON format"""
        return json.dumps(self.graph, indent=2)
        
    def export_to_csv(self) -> Tuple[str, str]:
        """Export the graph to CSV format (nodes and edges)"""
        # Create nodes CSV
        nodes_df = pd.DataFrame(self.graph['nodes'])
        nodes_csv = nodes_df.to_csv(index=False)
        
        # Create edges CSV
        edges_df = pd.DataFrame(self.graph['relationships'])
        edges_csv = edges_df.to_csv(index=False)
        
        return nodes_csv, edges_csv