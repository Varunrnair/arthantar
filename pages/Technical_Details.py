import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Technical Details - Arthantar",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Technical Details")
    st.subheader("Understanding the Technology Behind Arthantar")
    
    # Overview
    st.markdown("""
    ## Overview
    
    Arthantar is a sophisticated multi-layered approach to improve translation by utilizing a combination of gender identification, coreference resolution, and knowledge graph generation. The core of this approach relies on multiple technologies and models, each serving a specific function.
    """)
    
    # Create tabs for different components
    tab1, tab2, tab3, tab4 , tab5= st.tabs(["Flow","Coreference Resolution", "LLM Integration", "Knowledge Graph", "Translation Process"])
    
    with tab1:
        st.image("assets/mainimage.png", caption="Knowledge Graph Visualization", use_container_width=True)

    with tab2:
        # Example section at the start of the tab
        st.markdown("### Example of Coreference Resolution in Action")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original Text")
            st.info("""
            Sheela went to the market. She bought vegetables and fruits. 
            Later she gave some to her neighbor Rahul. He thanked her.
            """)
        
        with col2:
            st.markdown("#### Identified Clusters")
            st.success("""
            Cluster 1: ['Sheela', 'She', 'she', 'her']
            Cluster 2: ['Rahul', 'He']
            """)
        
        st.markdown("This example shows how coreference resolution identifies that 'Sheela', 'She', and 'her' refer to the same female entity, while 'Rahul' and 'He' refer to a male entity.")
        
        st.divider()
        
        st.markdown("""
        ### Coreference Resolution
        
        For gender identification, Arthantar uses the `FCoref` module, a coreference resolution tool that identifies gendered pronouns and assigns genders to entities based on context.
        
        #### How it works:
        
        1. The module analyzes the text and identifies clusters of related pronouns
        2. It uses predefined sets of male and female pronouns to determine the likely gender of each entity
        3. The system resolves ambiguous pronouns and determines gender accurately, with a focus on proper noun gender inference
        
        #### Example:
        
        In the sentence "Kiran is a good student and she goes to school", the coreference resolution identifies that "Kiran" and "she" refer to the same entity, and therefore Kiran is female.
        """)
        
        # Code snippet
        st.code("""
# Example of coreference resolution
from fastcoref import FCoref

coref_model = FCoref()
text = "Kiran is a good student and she goes to school."
preds = coref_model.predict(texts=[text])
clusters = preds[0].get_clusters()

# Output: [['Kiran', 'she']]
print(clusters)
        """, language="python")
    
    with tab3:
        # Example section at the start of the tab
        st.markdown("### Example of LLM Gender Identification")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### LLM Prompt")
            st.info("""
            Analyze the entity name 'Aarav' with type 'Person'.
            If it's a person, determine their likely gender based on context.
            Respond with exactly one word: 'male', 'female', or 'unknown'.
            For non-person entities, always respond with 'unknown'.
            """)
        
        with col2:
            st.markdown("#### LLM Response")
            st.success("male")
        
        st.markdown("In this example, the LLM correctly identifies 'Aarav' as a traditionally male Indian name when coreference resolution couldn't determine the gender.")
        
        st.divider()
        
        st.markdown("""
        ### Large Language Model Integration
        
        Arthantar incorporates an advanced large language model (LLM) using the Groq API. For gender identification when coreference resolution fails, it uses models like `llama-3.1-8b-8192` or `mixtral-8x7b-32768`.
        
        #### How it works:
        
        1. The LLM is employed as a backup to predict the likely gender of entities
        2. It analyzes contextual clues such as the name or role described
        3. If the coreference model doesn't provide a clear gender, the system sends a prompt to the LLM
        4. The LLM analyzes the entity name and its type and predicts its gender
        
        #### Example prompt:
        """)
        
        st.code("""
prompt = f\"\"\"
Analyze the entity name 'Kiran' with type 'Person'.
If it's a person, determine their likely gender based on context.
Respond with exactly one word: 'male', 'female', or 'unknown'.
For non-person entities, always respond with 'unknown'.
\"\"\"

response = llm.predict(prompt)
# Output might be: "female"
        """, language="python")
    
    with tab4:
        # Example section at the start of the tab
        st.markdown("### Example Knowledge Graph Visualization")
        
        st.markdown("""
        Below is a visual representation of a simple knowledge graph generated from the text:
        "John went to the store with his brother Michael, who was buying groceries."
        """)
        
        try:
            st.image("assets/kg1.png", caption="Knowledge Graph Visualization", use_container_width=True)
        except:
            st.warning("Knowledge graph image not found. Please add an image at 'assets/knowledge_graph_example.png' or update the path.")
            st.markdown("""
            **Note:** To add your own knowledge graph visualization:
            1. Create a folder named 'assets' in your project directory
            2. Add your knowledge graph image as 'knowledge_graph_example.png'
            3. Or modify the code to use your own image path
            """)
        
        st.divider()
        
        st.markdown("""
        ### Knowledge Graph Generation
        
        For knowledge graph generation, Arthantar uses the `LLMGraphTransformer` library, which facilitates the creation of a knowledge graph that encapsulates entities and their relationships.
        
        #### How it works:
        
        1. The knowledge graph is fed into the LLM, which processes the input text and extracts relevant information
        2. The graph includes entities, gender data, and relationships between them
        3. The knowledge graph is built by first converting the input text into graph documents
        4. Nodes and relationships are created using the transformer's output
        5. These nodes are enriched with gender information identified by coreference resolution or LLM
        
        #### Fallback mechanisms:
        
        If both the coreference and LLM fail to provide sufficient data, a more sophisticated backup method is employed using spaCy's natural language processing capabilities:
        
        1. Named entity recognition (NER) extracts entities like persons, organizations, and locations
        2. Dependency parsing establishes relationships through syntactic analysis
        3. The NetworkX library creates a directed graph where entities are connected based on syntactic relationships
        """)
        
        # Simple code example for knowledge graph generation
        st.code("""
# Example of knowledge graph generation
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes with gender information
G.add_node("Kiran", type="Person", gender="female")
G.add_node("school", type="Location", gender="unknown")

# Add relationships
G.add_edge("Kiran", "school", type="GOES_TO")

# Convert to dictionary format
graph = {
    'nodes': [{
        'id': n,
        'type': G.nodes[n]['type'],
        'gender': G.nodes[n]['gender']
    } for n in G.nodes()],
    'relationships': [{
        'source': u,
        'target': v,
        'type': G.edges[u, v]['type']
    } for u, v in G.edges()]
}
        """, language="python")
    
    with tab5:
        # Example section at the start of the tab
        st.markdown("### Example of Enhanced Translation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Standard Translation")
            st.info("""
            **English**: "Kiran is a good student and she goes to school."
            
            **Hindi (Standard)**: "किरण एक अच्छा छात्र है और वह स्कूल जाता है।"
            
            *Note: The gender is ambiguous and defaults to masculine form*
            """)
        
        with col2:
            st.markdown("#### Arthantar Enhanced Translation")
            st.success("""
            **English**: "Kiran is a good student and she goes to school."
            
            **Knowledge Graph**: Kiran (Person, female) --[GOES_TO]--> school (Location)
            
            **Hindi (Enhanced)**: "किरण एक अच्छी छात्रा है और वह स्कूल जाती है।"
            
            *Note: Correctly uses feminine form based on knowledge graph*
            """)
        
        st.markdown("This example demonstrates how the knowledge graph provides critical gender context that leads to a more accurate translation with proper gender agreement in Hindi.")
        
        st.divider()
        
        st.markdown("""
        ### Translation Process
        
        The knowledge graph plays a crucial role in providing contextual information to the translation process, ensuring that the translation model can handle gender nuances and understand the relationships between entities.
        
        #### Standard vs. Enhanced Translation:
        
        1. **Standard Translation**: Uses only the source text without additional context
        2. **Enhanced Translation**: Incorporates knowledge graph metadata to inform gender and relationship context
        
        #### Example of enhanced translation prompt:
        """)
        
        st.code("""
# Example of enhanced translation prompt
prompt = {
    "role": "user",
    "content": (
        "Translate to Hindi (Devanagari) using only the gender and relationship context "
        "of entities from metadata; reply with only the Hindi text: "
        "'Kiran is a good student and she goes to school.' "
        "Meta data: Knowledge Graph Structure: Nodes: Kiran (Type: Person, Gender: female), "
        "school (Type: Location, Gender: unknown). "
        "Relationships: Kiran --[GOES_TO]--> school."
    )
}
        """, language="python")
        
        st.markdown("""
        #### Multi-level Fallback System:
        
        The system implements multi-level fallback through exception handling:
        
        1. If the coreference model encounters errors, it defaults to the LLM
        2. If the LLM fails, it uses the spaCy-based graph generation
        3. If spaCy processing fails, it falls back to a basic entity extraction using capitalized words
        
        This ensures that, regardless of the failure point, a semantically meaningful graph is always available to provide contextual information for the translation model.
        """)
    
    
    # System architecture diagram
    st.markdown("""
    ## System Architecture
    
    The Arthantar system follows a multi-layered architecture with several components working together:
    
    1. **Input Processing Layer**: Handles text input and preprocessing
    2. **Coreference Resolution Layer**: Identifies entity references and gender information
    3. **Knowledge Graph Generation Layer**: Creates a structured representation of entities and relationships
    4. **Translation Layer**: Uses the knowledge graph to enhance translation quality
    
    Each layer has fallback mechanisms to ensure robustness and reliability.
    """)
    
    # Performance considerations
    st.markdown("""
    ## Performance Considerations
    
    ### Caching
    
    The Streamlit application implements caching to improve performance:
    
    1. Previously processed examples are stored in the session state
    2. Models are loaded lazily to reduce startup time
    3. Computationally expensive operations like graph generation are only performed when necessary
    
    ### API Usage
    
    The application is designed to minimize API calls:
    
    1. Coreference resolution is performed locally when possible
    2. LLM calls are only made when necessary (e.g., when coreference resolution fails)
    3. Results are cached to avoid redundant API calls
    """)
    
    # Future improvements
    st.markdown("""
    ## Future Improvements
    
    The Arthantar system could be enhanced in several ways:
    
    1. **Support for more languages**: Extend beyond Hindi to other languages with complex gender systems
    2. **Improved entity recognition**: Enhance the accuracy of entity and gender identification
    3. **More sophisticated knowledge graphs**: Incorporate additional relationship types and entity attributes
    4. **User feedback loop**: Allow users to correct gender assignments and improve the system over time
    5. **Offline mode**: Implement fully local models for environments without internet access
    """)

if __name__ == "__main__":
    main()