import streamlit as st
import os
from dotenv import load_dotenv
import time
from utils.translator import EnhancedTranslator
from utils.graph import KnowledgeGraphGenerator
from utils.visualizer import KnowledgeGraphVisualizer

# Load environment variables
load_dotenv()

# Check for API keys and provide setup instructions if missing
def check_api_keys():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key not found. Please add it to your .env file.")
        st.code("GROQ_API_KEY=your_api_key_here", language="text")
        return False
    return True

# Initialize session state for caching
if 'cached_examples' not in st.session_state:
    st.session_state.cached_examples = {}
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'graph_generator' not in st.session_state:
    st.session_state.graph_generator = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None

# Initialize components if API keys are available
def initialize_components():
    if check_api_keys() and st.session_state.translator is None:
        with st.spinner("Initializing components..."):
            groq_api_key = os.getenv("GROQ_API_KEY")
            st.session_state.translator = EnhancedTranslator(groq_api_key)
            st.session_state.graph_generator = KnowledgeGraphGenerator(groq_api_key)
            st.session_state.visualizer = KnowledgeGraphVisualizer()
        st.success("Components initialized successfully!")

# Main app
def main():
    st.set_page_config(
        page_title="Arthantar - Contextual Translation",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌐 Arthantar")
    st.subheader("Contextual Translation with Coreference Analysis, LLMs, and Knowledge Graphs")
    
    # Initialize components
    initialize_components()
    
    # Main page content
    st.markdown("""
    ## Welcome to Arthantar
    
    Arthantar is a sophisticated multi-layered approach to improve translation by utilizing:
    
    - **Coreference Resolution**: Identifies gendered pronouns and assigns genders to entities
    - **Large Language Models**: Enhances context understanding and gender identification
    - **Knowledge Graphs**: Captures entity relationships and contextual information
    
    This application demonstrates how these technologies work together to provide more accurate translations.
    """)
    
    # Quick demo section
    st.header("Quick Demo")
    
    # Cached examples
    st.subheader("Examples")
    example_texts = {
        "Example 1": "Kiran is a good student and she goes to school.",
        "Example 2": "Kiran is a good student. Sita is his science teacher, and he is Kiran's favourite teacher.",
        "Example 3": "John gave Mary a book because she loves reading."
    }
    
    selected_example = st.selectbox("Select an example", list(example_texts.keys()))
    
    if selected_example:
        example_text = example_texts[selected_example]
        st.text_area("Source Text", example_text, height=100)
        
        if st.button("Translate"):
            if check_api_keys():
                # Check if this example is already cached
                if selected_example in st.session_state.cached_examples:
                    result = st.session_state.cached_examples[selected_example]
                    st.success("Successful")
                else:
                    with st.spinner("Generating knowledge graph and translation..."):
                        # Generate knowledge graph
                        graph = st.session_state.graph_generator.create_graph_from_text(example_text)
                        
                        # Get standard translation
                        standard_translation = st.session_state.translator.get_standard_translation(example_text)
                        
                        # Get enhanced translation
                        enhanced_translation = st.session_state.translator.get_enhanced_translation(example_text, graph)
                        
                        # Cache the results
                        result = {
                            "graph": graph,
                            "standard_translation": standard_translation,
                            "enhanced_translation": enhanced_translation
                        }
                        st.session_state.cached_examples[selected_example] = result
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Standard Translation")
                    st.text(result["standard_translation"])
                
                with col2:
                    st.subheader("Enhanced Translation")
                    st.text(result["enhanced_translation"])
                
                # Display knowledge graph
                st.subheader("Knowledge Graph")
                st.session_state.visualizer.display_graph(result["graph"], example_text)
    
    # Navigation to other pages
    st.markdown("---")
    st.markdown("""
    ## Explore More
    
    Navigate to the other pages to explore different aspects of Arthantar:
    
    - **Live Demo**: Try your own text for translation
    - **Knowledge Graph Explorer**: Visualize and explore knowledge graphs in detail
    - **Technical Details**: Learn about the technology behind Arthantar
    """)

if __name__ == "__main__":
    main()