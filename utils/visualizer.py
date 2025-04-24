import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
import io
import base64

class KnowledgeGraphVisualizer:
    def __init__(self):
        """Initialize the visualizer with color schemes"""
        self.gender_colors = {
            'male': '#ADD8E6',    # Light blue
            'female': '#FFB6C1',  # Light pink
            'unknown': '#D3D3D3'  # Light gray
        }

        self.type_markers = {
            'Person': 'o',      # Circle
            'PERSON': 'o',      # Circle
            'Location': 's',    # Square
            'GPE': 's',         # Square for geo-political entities
            'Organization': '^', # Triangle
            'ORG': '^',         # Triangle
            'Entity': 'D'       # Diamond (default)
        }
        
        self.type_colors = {
            'Person': '#90EE90',  # Light green
            'PERSON': '#90EE90',  # Light green
            'Location': '#FFD700', # Gold
            'GPE': '#FFD700',     # Gold
            'Organization': '#FFA07A', # Light salmon
            'ORG': '#FFA07A',     # Light salmon
            'Entity': '#E6E6FA'   # Lavender (default)
        }

    def create_visualization_figure(self, graph: Dict, title: str = "Knowledge Graph", 
                                   layout: str = "spring", node_size: int = 2000,
                                   show_labels: bool = True) -> plt.Figure:
        """Create a matplotlib figure with the knowledge graph visualization"""
        # Create a new directed graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in graph['nodes']:
            G.add_node(
                node['id'],
                gender=node.get('gender', 'unknown'),
                type=node.get('type', 'Entity')
            )

        # Add edges with relationship types
        for rel in graph['relationships']:
            G.add_edge(
                rel['source'],
                rel['target'],
                relationship=rel['type']
            )

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create the layout based on user selection
        if layout == "spring":
            pos = nx.spring_layout(G, k=0.3, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "planar":
            try:
                pos = nx.planar_layout(G)
            except:
                pos = nx.spring_layout(G)  # Fallback if graph is not planar
        elif layout == "random":
            pos = nx.random_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Draw nodes by type
        for node_type in set(nx.get_node_attributes(G, 'type').values()):
            node_list = [node for node, attr in G.nodes(data=True)
                        if attr.get('type') == node_type]

            if node_list:
                # Use gender for color and type for shape
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=node_list,
                    node_color=[self.gender_colors[G.nodes[node]['gender']]
                              for node in node_list],
                    node_size=node_size,
                    node_shape=self.type_markers.get(node_type, 'o'),
                    label=node_type,
                    edgecolors='black',
                    alpha=0.8
                )

        # Draw edges
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, width=1.5, alpha=0.7)

        # Add labels if requested
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        # Add legend for node types
        legend_elements = [plt.Line2D([0], [0], marker=self.type_markers.get(node_type, 'o'),
                                    color='w', markerfacecolor='gray', markersize=10,
                                    label=node_type)
                         for node_type in set(nx.get_node_attributes(G, 'type').values())]

        # Add legend for genders
        legend_elements.extend([plt.Line2D([0], [0], marker='o',
                                         color='w', markerfacecolor=color, markersize=10,
                                         label=gender)
                              for gender, color in self.gender_colors.items()])

        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        # Set title and layout
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        return fig

    def get_image_bytes(self, fig: plt.Figure) -> bytes:
        """Convert a matplotlib figure to bytes for download"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
        
    def create_subgraph_visualization(self, graph: Dict, focus_entity: str, 
                                     depth: int = 1) -> plt.Figure:
        """Create a visualization focused on a specific entity"""
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph['nodes']:
            G.add_node(
                node['id'],
                gender=node.get('gender', 'unknown'),
                type=node.get('type', 'Entity')
            )
            
        # Add edges
        for rel in graph['relationships']:
            G.add_edge(
                rel['source'],
                rel['target'],
                relationship=rel['type']
            )
            
        # Create subgraph centered on focus entity
        if focus_entity not in G:
            # If entity not found, return empty graph
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"Entity '{focus_entity}' not found in graph", 
                   ha='center', va='center')
            ax.axis('off')
            return fig
            
        # Get nodes within specified depth
        nodes = {focus_entity}
        current_nodes = {focus_entity}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(G.predecessors(node))
                next_nodes.update(G.successors(node))
            current_nodes = next_nodes - nodes
            nodes.update(current_nodes)
            
        # Create subgraph
        subgraph = G.subgraph(nodes)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=0.3, iterations=50)
        
        # Draw focus node with special highlighting
        nx.draw_networkx_nodes(
            subgraph, pos,
            nodelist=[focus_entity],
            node_color='yellow',
            node_size=3000,
            edgecolors='red',
            linewidths=3
        )
        
        # Draw other nodes by type
        for node_type in set(nx.get_node_attributes(subgraph, 'type').values()):
            node_list = [node for node, attr in subgraph.nodes(data=True)
                        if attr.get('type') == node_type and node != focus_entity]
            
            if node_list:
                nx.draw_networkx_nodes(
                    subgraph, pos,
                    nodelist=node_list,
                    node_color=[self.gender_colors[subgraph.nodes[node]['gender']]
                              for node in node_list],
                    node_size=2000,
                    node_shape=self.type_markers.get(node_type, 'o'),
                    edgecolors='black'
                )
                
        # Draw edges
        edge_labels = nx.get_edge_attributes(subgraph, 'relationship')
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=8)
        
        # Set title and layout
        plt.title(f"Subgraph centered on '{focus_entity}' (depth={depth})")
        plt.axis('off')
        plt.tight_layout()
        
        return fig