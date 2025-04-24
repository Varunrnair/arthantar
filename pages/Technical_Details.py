import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.translator import EnhancedKnowledgeGraphGenerator
import networkx as nx

# Set page configuration
st.set_page_config(
    page_title="Technical Details - Arthantar",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Technical Details")
st.subheader("Understanding the Arthantar System")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "System Architecture", 
    "Coreference Resolution", 
    "Knowledge Graph Generation",
    "Translation Process"
])

with tab1:
    st.header("System Architecture")
    
    st.markdown("""
    ### Multi-Layered Approach
    
    Arthantar implements a sophisticated multi-layered approach to improve translation by utilizing:
    
    1. **Gender Identification Layer**
       - Primary: FCoref module for coreference resolution
       - Backup: LLM-based gender prediction using Groq API
       
    2. **Knowledge Graph Generation Layer**
       - Primary: LLMGraphTransformer with Groq API
       - Backup: spaCy-based entity and relationship extraction
       - Fallback: Basic entity extraction using capitalized words
       
    3. **Translation Enhancement Layer**
       - Contextual prompt generation with knowledge graph metadata
       - Gender and relationship-aware translation
    
    ### System Components
    """)
    
    # Create a diagram of the system architecture
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ```mermaid
        graph TD;
            A["Input Text"] --> B["Coreference Resolution (FCoref)"]
            B --> C["Gender Identification"]
            A --> D["Knowledge Graph Generation"]
            C --> D
            D --> E["Contextual Prompt Generation"]
            E --> F["Enhanced Translation (Groq LLM)"]
            F --> G["Final Translation"]
            
            B -- "Failure" --> H["LLM Gender Prediction"]
            H --> D
            D -- "Failure" --> I["spaCy Backup Graph"]
            I --> E
            I -- "Failure" --> J["Basic Entity Extraction"]
            J --> E
        ```
        """)
    
    with col2:
        st.markdown("""
        ### Key Technologies
        
        - **FCoref**: State-of-the-art coreference resolution
        - **Groq API**: LLM integration for various tasks
        - **LangChain**: Framework for LLM applications
        - **spaCy**: NLP toolkit for backup processing
        - **NetworkX**: Graph operations and analysis
        - **Streamlit**: Interactive web interface
        """)
    
    st.markdown("""
    ### Fallback Mechanisms
    
    The system implements multiple fallback mechanisms to ensure robustness:
    
    1. If coreference resolution fails → Use LLM for gender prediction
    2. If LLM graph generation fails → Use spaCy-based graph generation
    3. If spaCy processing fails → Use basic entity extraction
    
    This ensures that even in challenging scenarios, the system can still provide useful translations with contextual awareness.
    """)

with tab2:
    st.header("Coreference Resolution")
    
    st.markdown("""
    ### How Coreference Resolution Works
    
    Coreference resolution is the task of finding all expressions that refer to the same entity in a text. In Arthantar, we use the FCoref module, which identifies clusters of related pronouns and assigns genders to entities based on context.
    
    #### Example:
    
    In the sentence "**Kiran** is a good student and **she** goes to school", coreference resolution identifies that "Kiran" and "she" refer to the same entity, and therefore Kiran is likely female.
    
    ### Gender Identification Process
    
    1. The text is analyzed to identify clusters of related mentions
    2. Each cluster is checked for gendered pronouns (he/him/his or she/her/hers)
    3. If a cluster contains gendered pronouns, all entities in that cluster are assigned the corresponding gender
    4. For entities without clear gender indicators, the LLM is used as a backup
    
    ### Challenges in Coreference Resolution
    
    - **Ambiguous Pronouns**: When pronouns could refer to multiple entities
    - **Implicit References**: When entities are referenced without explicit pronouns
    - **Cross-Cultural Names**: Names that may be used for different genders in different cultures
    
    Arthantar addresses these challenges through its multi-layered approach and fallback mechanisms.
    """)
    
    # Add a simple example visualization
    st.subheader("Example Coreference Clusters")
    
    example_text = "Kiran is a good student. Sita is his science teacher, and he is Kiran's favorite teacher."
    
    if st.button("Run Coreference Example"):
        if 'GROQ_API_KEY' in st.session_state and st.session_state['GROQ_API_KEY']:
            try:
                kg_generator = EnhancedKnowledgeGraphGenerator(st.session_state['GROQ_API_KEY'])
                gender_map = kg_generator.identify_genders_coref(example_text)
                
                # Display clusters
                preds = kg_generator.coref_model.predict(texts=[example_text])
                clusters = preds[0].get_clusters()
                
                for i, cluster in enumerate(clusters):
                    st.markdown(f"**Cluster {i+1}:** {' | '.join(cluster)}")
                
                # Display gender map
                if gender_map:
                    st.subheader("Identified Genders")
                    gender_df = pd.DataFrame(
                        [(entity, gender) for entity, gender in gender_map.items()],
                        columns=["Entity", "Gender"]
                    )
                    st.dataframe(gender_df)
                else:
                    st.info("No gender information identified from coreference.")
            except Exception as e:
                st.error(f"Error running coreference example: {str(e)}")
        else:
            st.warning("Please enter your Groq API key on the home page to run this example.")

with tab3:
    st.header("Knowledge Graph Generation")
    
    st.markdown("""
    ### Knowledge Graph Components
    
    A knowledge graph represents entities and their relationships in a structured format. In Arthantar, our knowledge graphs contain:
    
    - **Nodes**: Representing entities with attributes like type and gender
    - **Relationships**: Representing connections between entities
    
    ### Generation Process
    
    1. **Primary Method**: Using LLMGraphTransformer with Groq API
       - Text is processed by the LLM to extract entities and relationships
       - Gender information is added from coreference resolution
       - A structured graph is created using NetworkX
    
    2. **Backup Method**: Using spaCy NLP
       - Named Entity Recognition (NER) identifies entities
       - Dependency parsing identifies relationships
       - Gender information is added from coreference or LLM prediction
    
    3. **Fallback Method**: Basic entity extraction
       - Capitalized words are treated as entities
       - Sequential relationships are created
       - Gender information is added where available
    
    ### Knowledge Graph Applications
    
    - **Context Enhancement**: Provides structural context for translation
    - **Gender Tracking**: Ensures consistent gender usage across languages
    - **Relationship Preservation**: Maintains semantic relationships in translation
    """)
    
    # Add a simple example visualization
    st.subheader("Example Knowledge Graph")
    
    example_text = "Kiran is a good student. Sita is his science teacher, and he is Kiran's favorite teacher."
    
    if st.button("Generate Example Graph"):
        if 'GROQ_API_KEY' in st.session_state and st.session_state['GROQ_API_KEY']:
            try:
                kg_generator = EnhancedKnowledgeGraphGenerator(st.session_state['GROQ_API_KEY'])
                graph = kg_generator.create_graph_from_text(example_text)
                
                # Create NetworkX graph for visualization
                G = nx.DiGraph()
                
                # Add nodes
                for node in graph['nodes']:
                    G.add_node(node['id'], type=node['type'], gender=node['gender'])
                
                # Add edges
                for rel in graph['relationships']:
                    G.add_edge(rel['source'], rel['target'], type=rel['type'])
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(G, seed=42)
                
                # Node colors based on gender
                gender_colors = {
                    'male': '#ADD8E6',    # Light blue
                    'female': '#FFB6C1',  # Light pink
                    'unknown': '#D3D3D3'  # Light gray
                }
                
                node_colors = [gender_colors[G.nodes[n]['gender']] for n in G.nodes()]
                
                nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                       node_size=2000, font_size=10, font_weight='bold')
                
                edge_labels = {(u, v): G.edges[u, v]['type'] for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                
                plt.title("Example Knowledge Graph")
                plt.axis('off')
                st.pyplot(fig)
                
                # Display graph data
                st.subheader("Graph Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Nodes**")
                    nodes_df = pd.DataFrame(graph['nodes'])
                    st.dataframe(nodes_df)
                
                with col2:
                    st.markdown("**Relationships**")
                    if graph['relationships']:
                        rels_df = pd.DataFrame(graph['relationships'])
                        st.dataframe(rels_df)
                    else:
                        st.info("No relationships found in the graph.")
                
            except Exception as e:
                st.error(f"Error generating example graph: {str(e)}")
        else:
            st.warning("Please enter your Groq API key on the home page to run this example.")

with tab4:
    st.header("Translation Process")
    
    st.markdown("""
    ### Contextual Translation
    
    Arthantar enhances translation by incorporating contextual information from the knowledge graph:
    
    1. **Prompt Generation**
       - The knowledge graph is converted to a metadata string
       - This metadata includes entity types, genders, and relationships
       - A specialized prompt is created for the LLM
    
    2. **Translation with Context**
       - The LLM uses the knowledge graph metadata to inform translation
       - Gender information ensures proper gender agreement in the target language
       - Relationship information preserves semantic connections
    
    3. **Advantages Over Standard Translation**
       - **Gender Accuracy**: Correctly handles gendered pronouns and agreements
       - **Contextual Awareness**: Understands entity relationships
       - **Semantic Preservation**: Maintains meaning across languages
    
    ### Example Translation Process
    """)
    
    example_text = "Kiran is a good student. Sita is his science teacher, and he is Kiran's favorite teacher."
    
    if st.button("Show Translation Process"):
        if 'GROQ_API_KEY' in st.session_state and st.session_state['GROQ_API_KEY']:
            try:
                kg_generator = EnhancedKnowledgeGraphGenerator(st.session_state['GROQ_API_KEY'])
                
                # Step 1: Generate knowledge graph
                with st.spinner("Generating knowledge graph..."):
                    graph = kg_generator.create_graph_from_text(example_text)
                
                # Step 2: Generate translation prompt
                with st.spinner("Generating translation prompt..."):
                    prompt = kg_generator.generate_translation_prompt(example_text, graph)
                
                # Step 3: Translate
                with st.spinner("Translating..."):
                    translation = kg_generator.translate_text(prompt)
                    
                    # Also get standard translation for comparison
                    standard_prompt = {
                        "role": "user",
                        "content": f"Translate this to Hindi and send only the translated part: {example_text}"
                    }
                    standard_translation = kg_generator.translate_text(standard_prompt)
                
                # Display results
                st.subheader("Translation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Contextual Translation**")
                    st.markdown(f"**{translation}**")
                
                with col2:
                    st.markdown("**Standard Translation**")
                    st.markdown(f"**{standard_translation}**")
                
                # Display prompt
                with st.expander("View Translation Prompt"):
                    st.code(prompt["content"], language="text")
                
            except Exception as e:
                st.error(f"Error in translation process: {str(e)}")
        else:
            st.warning("Please enter your Groq API key on the home page to run this example.")
    
    st.markdown("""
    ### Future Improvements
    
    - **Multi-language Support**: Extend beyond Hindi to other languages
    - **Enhanced Entity Recognition**: Improve identification of complex entities
    - **Relationship Extraction**: Develop more sophisticated relationship detection
    - **Performance Optimization**: Reduce processing time for real-time applications
    """)

# Footer
st.markdown("---")
st.markdown("Arthantar - Contextual Translation System")