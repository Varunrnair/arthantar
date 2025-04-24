import streamlit as st
import pandas as pd
import networkx as nx
from utils.visualizer import KnowledgeGraphVisualizer
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Knowledge Graph Explorer - Arthantar",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Knowledge Graph Explorer")
st.subheader("Visualize and explore the knowledge graph")

# Check if we have a graph to display
if 'last_graph' not in st.session_state:
    st.info("No knowledge graph available. Please run a translation on the Live Demo page first.")
    st.stop()

# Display the last text and translation
if 'last_text' in st.session_state and 'last_translation' in st.session_state:
    with st.expander("Last Translation", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Text")
            st.write(st.session_state['last_text'])
        with col2:
            st.subheader("Translation")
            st.write(st.session_state['last_translation'])

# Get the graph from session state
graph = st.session_state['last_graph']

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Visual Explorer", "Data View", "Network Analysis"])

with tab1:
    st.header("Knowledge Graph Visualization")
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    with col1:
        layout_algorithm = st.selectbox(
            "Layout Algorithm",
            ["spring", "circular", "kamada_kawai", "planar", "random", "shell", "spectral"],
            index=0
        )
    with col2:
        node_size = st.slider("Node Size", min_value=100, max_value=3000, value=2000, step=100)
    with col3:
        show_labels = st.checkbox("Show Labels", value=True)
    
    # Create and display visualization
    visualizer = KnowledgeGraphVisualizer()
    fig = visualizer.create_visualization_figure(
        graph, 
        title="Knowledge Graph Visualization",
        layout=layout_algorithm,
        node_size=node_size,
        show_labels=show_labels
    )
    st.pyplot(fig)
    
    # Option to download the visualization
    btn = st.download_button(
        label="Download Visualization",
        data=visualizer.get_image_bytes(fig),
        file_name="knowledge_graph.png",
        mime="image/png"
    )

with tab2:
    st.header("Graph Data")
    
    # Display nodes
    st.subheader("Nodes")
    if graph['nodes']:
        node_df = pd.DataFrame(graph['nodes'])
        st.dataframe(node_df, use_container_width=True)
    else:
        st.info("No nodes found in the graph.")
    
    # Display relationships
    st.subheader("Relationships")
    if graph['relationships']:
        rel_df = pd.DataFrame(graph['relationships'])
        st.dataframe(rel_df, use_container_width=True)
    else:
        st.info("No relationships found in the graph.")
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export as JSON"):
            import json
            import base64
            
            # Convert graph to JSON
            json_str = json.dumps(graph, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="knowledge_graph.json">Download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("Export as CSV"):
            import io
            import base64
            
            # Create buffer
            buffer = io.StringIO()
            
            # Write nodes
            buffer.write("# Nodes\n")
            buffer.write("id,type,gender\n")
            for node in graph['nodes']:
                buffer.write(f"{node['id']},{node['type']},{node['gender']}\n")
            
            # Write relationships
            buffer.write("\n# Relationships\n")
            buffer.write("source,target,type\n")
            for rel in graph['relationships']:
                buffer.write(f"{rel['source']},{rel['target']},{rel['type']}\n")
            
            # Create download link
            b64 = base64.b64encode(buffer.getvalue().encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="knowledge_graph.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

with tab3:
    st.header("Network Analysis")
    
    # Create a NetworkX graph from the data
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph['nodes']:
        G.add_node(node['id'], type=node['type'], gender=node['gender'])
    
    # Add edges
    for rel in graph['relationships']:
        G.add_edge(rel['source'], rel['target'], type=rel['type'])
    
    # Display graph metrics
    st.subheader("Graph Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", len(G.nodes()))
    col2.metric("Edges", len(G.edges()))
    col3.metric("Density", round(nx.density(G), 4))
    
    try:
        col4.metric("Diameter", nx.diameter(G))
    except:
        col4.metric("Diameter", "N/A")
    
    # Node centrality
    st.subheader("Node Centrality")
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
    except:
        betweenness_centrality = {node: 0 for node in G.nodes()}
    
    # Create DataFrame
    centrality_df = pd.DataFrame({
        'Node': list(G.nodes()),
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'In-Degree Centrality': [in_degree_centrality[node] for node in G.nodes()],
        'Out-Degree Centrality': [out_degree_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()]
    })
    
    # Sort by total centrality
    centrality_df['Total Centrality'] = (
        centrality_df['Degree Centrality'] + 
        centrality_df['Betweenness Centrality']
    )
    centrality_df = centrality_df.sort_values('Total Centrality', ascending=False)
    
    # Display centrality table
    st.dataframe(centrality_df, use_container_width=True)
    
    # Visualize centrality
    st.subheader("Centrality Visualization")
    
    centrality_metric = st.selectbox(
        "Centrality Measure",
        ["Degree Centrality", "In-Degree Centrality", "Out-Degree Centrality", "Betweenness Centrality"]
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the centrality values
    metric_col = centrality_metric
    centrality_values = centrality_df[metric_col].tolist()
    nodes = centrality_df['Node'].tolist()
    
    # Create bar chart
    bars = ax.barh(nodes, centrality_values)
    
    # Add labels
    ax.set_xlabel(centrality_metric)
    ax.set_title(f"{centrality_metric} by Node")
    
    # Color by gender
    gender_colors = {
        'male': '#ADD8E6',    # Light blue
        'female': '#FFB6C1',  # Light pink
        'unknown': '#D3D3D3'  # Light gray
    }
    
    for i, node in enumerate(nodes):
        gender = G.nodes[node].get('gender', 'unknown')
        bars[i].set_color(gender_colors.get(gender, '#D3D3D3'))
    
    # Display figure
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Arthantar - Contextual Translation System")