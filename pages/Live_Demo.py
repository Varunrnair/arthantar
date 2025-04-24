import streamlit as st
import pandas as pd
from utils.translator import EnhancedKnowledgeGraphGenerator
from utils.visualizer import KnowledgeGraphVisualizer

# Set page configuration
st.set_page_config(
    page_title="Live Demo - Arthantar",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Live Demo")
st.subheader("See Arthantar in Action")

# Check if API key is available
if not st.session_state.get('GROQ_API_KEY', ''):
    st.warning("Please enter your Groq API key on the home page to use this demo.")
    st.stop()

# Input section
st.header("Input Text")
input_text = st.text_area(
    "Enter text to analyze and translate",
    value="Kiran is a good student. Sita is his science teacher, and he is Kiran's favorite teacher.",
    height=150
)

col1, col2 = st.columns(2)
source_lang = col1.selectbox("Source Language", ["English"], index=0)
target_lang = col2.selectbox("Target Language", ["Hindi", "Spanish", "French", "German"], index=0)

# Analysis options
st.header("Analysis Options")
show_coref = st.checkbox("Show coreference clusters", value=True)
show_gender = st.checkbox("Show gender identification", value=True)
show_graph = st.checkbox("Show knowledge graph visualization", value=True)
compare_translations = st.checkbox("Compare with standard translation", value=True)

if st.button("Analyze and Translate", type="primary"):
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
        st.stop()
        
    with st.spinner("Processing..."):
        try:
            # Initialize translator
            kg_generator = EnhancedKnowledgeGraphGenerator(st.session_state['GROQ_API_KEY'])
            
            # Step 1: Identify genders and coreference
            if show_coref:
                with st.expander("Coreference Analysis", expanded=True):
                    st.subheader("Coreference Clusters")
                    gender_map = kg_generator.identify_genders_coref(input_text)
                    
                    # Display clusters
                    preds = kg_generator.coref_model.predict(texts=[input_text])
                    clusters = preds[0].get_clusters()
                    
                    for i, cluster in enumerate(clusters):
                        st.markdown(f"**Cluster {i+1}:** {' | '.join(cluster)}")
                    
                    if gender_map:
                        st.subheader("Identified Genders")
                        gender_df = pd.DataFrame(
                            [(entity, gender) for entity, gender in gender_map.items()],
                            columns=["Entity", "Gender"]
                        )
                        st.dataframe(gender_df)
                    else:
                        st.info("No gender information identified from coreference.")
            
            # Step 2: Generate knowledge graph
            graph = kg_generator.create_graph_from_text(input_text)
            st.session_state['last_graph'] = graph
            
            if show_graph:
                with st.expander("Knowledge Graph", expanded=True):
                    st.subheader("Knowledge Graph Visualization")
                    visualizer = KnowledgeGraphVisualizer()
                    fig = visualizer.create_visualization_figure(graph, f"Knowledge Graph")
                    st.pyplot(fig)
                    
                    # Also show nodes and relationships
                    col1, col2 = st.columns(2)
                    
                    # Display nodes
                    with col1:
                        st.subheader("Nodes")
                        nodes_df = pd.DataFrame(graph['nodes'])
                        st.dataframe(nodes_df)
                    
                    # Display relationships
                    with col2:
                        st.subheader("Relationships")
                        if graph['relationships']:
                            rel_df = pd.DataFrame(graph['relationships'])
                            st.dataframe(rel_df)
                        else:
                            st.info("No relationships identified.")
            
            # Step 3: Generate translation prompt and translate
            with st.expander("Translation Results", expanded=True):
                st.subheader("Translation Results")
                
                # Generate contextual translation
                prompt = kg_generator.generate_translation_prompt(input_text, graph, target_lang.lower())
                contextual_translation = kg_generator.translate_text(prompt)
                
                # Generate standard translation for comparison
                standard_translation = None
                if compare_translations:
                    standard_prompt = {
                        "role": "user",
                        "content": f"Translate this to {target_lang} and send only the translated part: {input_text}"
                    }
                    standard_translation = kg_generator.translate_text(standard_prompt)
                
                # Display translations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Contextual Translation")
                    st.markdown(f"**{contextual_translation}**")
                
                if compare_translations and standard_translation:
                    with col2:
                        st.markdown("#### Standard Translation")
                        st.markdown(f"**{standard_translation}**")
                        
                # Store in session state
                st.session_state['last_text'] = input_text
                st.session_state['last_translation'] = contextual_translation
                if compare_translations and standard_translation:
                    st.session_state['last_standard_translation'] = standard_translation
                
                st.success("Analysis and translation complete!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your API key and try again.")

# Footer
st.markdown("---")
st.markdown("Arthantar - Contextual Translation System")