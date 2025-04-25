import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
import io

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
            'GPE': 's',         # Square for locations
            'Organization': '^', # Triangle
            'ORG': '^',         # Triangle for organizations
            'Entity': 'D'       # Diamond (default)
        }
    
    def create_visualization(self, graph: Dict, title: str = "Knowledge Graph Visualization"):
        """Create a visualization of the knowledge graph and return the figure"""
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
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw nodes
        for node_type in set(nx.get_node_attributes(G, 'type').values()):
            node_list = [node for node, attr in G.nodes(data=True)
                        if attr.get('type') == node_type]
            
            if node_list:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=node_list,
                    node_color=[self.gender_colors[G.nodes[node]['gender']]
                              for node in node_list],
                    node_size=2000,
                    node_shape=self.type_markers.get(node_type, 'o'),
                    label=node_type,
                    ax=ax
                )
        
        # Draw edges
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
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
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Set title and layout
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def display_graph(self, graph: Dict, text: str = ""):
        """Display the knowledge graph in Streamlit"""
        if not graph or not graph.get('nodes'):
            st.warning("No graph data available to visualize.")
            return
        
        # Create the visualization
        title = f"Knowledge Graph" if not text else f"Knowledge Graph for: {text[:50]}..."
        fig = self.create_visualization(graph, title)
        
        # Display the graph
        st.pyplot(fig)
        
        # Display graph details
        with st.expander("View Graph Details"):
            st.subheader("Nodes")
            for node in graph['nodes']:
                st.write(f"- **{node['id']}** (Type: {node['type']}, Gender: {node['gender']})")
            
            st.subheader("Relationships")
            for rel in graph['relationships']:
                st.write(f"- **{rel['source']}** --[{rel['type']}]--> **{rel['target']}**")