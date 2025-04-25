import streamlit as st
import os
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Knowledge Graph Explorer - Arthantar",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Knowledge Graph Explorer")
    st.subheader("Visualize and analyze knowledge graphs")
    
    # Check if components are initialized
    if 'graph_generator' not in st.session_state or st.session_state.graph_generator is None:
        st.warning("Components not initialized. Please return to the home page to initialize.")
        if st.button("Initialize Components"):
            st.experimental_rerun()
        return
    
    # Tabs for different exploration options
    tab1, tab2, tab3 = st.tabs(["Generate New Graph", "Explore Cached Graphs", "Graph Analysis"])
    
    with tab1:
        st.markdown("### Generate a New Knowledge Graph")
        
        # Input text
        input_text = st.text_area("Enter text to generate a knowledge graph", height=150)
        
        if st.button("Generate Graph") and input_text:
            with st.spinner("Generating knowledge graph..."):
                # Generate the graph
                graph = st.session_state.graph_generator.create_graph_from_text(input_text)
                
                # Store in session state for analysis
                if 'current_graph' not in st.session_state:
                    st.session_state.current_graph = {}
                st.session_state.current_graph = {
                    'text': input_text,
                    'graph': graph
                }
                
                # Display the graph
                st.session_state.visualizer.display_graph(graph, input_text)
                
                # Show raw graph data
                with st.expander("View Raw Graph Data"):
                    st.json(graph)
    
    with tab2:
        st.markdown("### Explore Cached Graphs")
        
        # Check if there are cached examples
        if 'cached_examples' not in st.session_state or not st.session_state.cached_examples:
            st.info("No cached graphs available. Generate some graphs from the home page or the Generate New Graph tab.")
        else:
            # Select from cached examples
            example_keys = list(st.session_state.cached_examples.keys())
            selected_example = st.selectbox("Select a cached example", example_keys)
            
            if selected_example:
                example_data = st.session_state.cached_examples[selected_example]
                example_text = example_data.get('text', selected_example)
                graph = example_data.get('graph')
                
                if graph:
                    # Display the graph
                    st.session_state.visualizer.display_graph(graph, example_text)
                    
                    # Show translations if available
                    if 'standard_translation' in example_data and 'enhanced_translation' in example_data:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Standard Translation")
                            st.text(example_data["standard_translation"])
                        
                        with col2:
                            st.subheader("Enhanced Translation")
                            st.text(example_data["enhanced_translation"])
    
    with tab3:
        st.markdown("### Graph Analysis")
        
        if 'current_graph' not in st.session_state or not st.session_state.current_graph:
            st.info("No graph selected for analysis. Generate a new graph or select a cached one.")
        else:
            graph = st.session_state.current_graph.get('graph')
            text = st.session_state.current_graph.get('text', '')
            
            if graph:
                # Display the graph
                st.session_state.visualizer.display_graph(graph, text)
                
                # Graph metrics
                st.subheader("Graph Metrics")
                
                # Create NetworkX graph for analysis
                G = nx.DiGraph()
                
                # Add nodes
                for node in graph['nodes']:
                    G.add_node(node['id'], **node)
                
                # Add edges
                for rel in graph['relationships']:
                    G.add_edge(rel['source'], rel['target'], type=rel['type'])
                
                # Calculate metrics
                metrics = {
                    "Number of Nodes": len(G.nodes()),
                    "Number of Edges": len(G.edges()),
                    "Density": nx.density(G),
                    "Is Connected": nx.is_weakly_connected(G),
                }
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    for key, value in list(metrics.items())[:2]:
                        st.metric(key, value)
                
                with col2:
                    for key, value in list(metrics.items())[2:]:
                        st.metric(key, value)
                
                # Node centrality
                st.subheader("Node Centrality")
                
                if len(G.nodes()) > 0:
                    try:
                        # Calculate centrality measures
                        degree_centrality = nx.degree_centrality(G)
                        in_degree_centrality = nx.in_degree_centrality(G)
                        out_degree_centrality = nx.out_degree_centrality(G)
                        
                        # Create a DataFrame for display
                        centrality_data = []
                        for node in G.nodes():
                            centrality_data.append({
                                "Node": node,
                                "Degree Centrality": round(degree_centrality[node], 3),
                                "In-Degree Centrality": round(in_degree_centrality[node], 3),
                                "Out-Degree Centrality": round(out_degree_centrality[node], 3)
                            })
                        
                        # Display as a table
                        st.dataframe(centrality_data)
                    except Exception as e:
                        st.error(f"Error calculating centrality: {str(e)}")
                else:
                    st.info("Not enough nodes to calculate centrality.")

if __name__ == "__main__":
    main()