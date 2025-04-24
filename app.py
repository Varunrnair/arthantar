import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import login
from utils.translator import EnhancedKnowledgeGraphGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

load_dotenv() 
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
# login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: Token not loaded from .env file")
else:
    print("Token loaded successfully")

# Set page configuration
st.set_page_config(
    page_title="Arthantar - Contextual Translation System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title and description
st.title("Arthantar")
st.subheader("Contextual Translation with Coreference Analysis & Knowledge Graphs")

st.markdown("""
This application demonstrates an advanced translation system that preserves gender context and entity relationships 
through coreference resolution and knowledge graph generation.

### Key Features:
- **Gender Identification** through coreference resolution and LLM backup
- **Entity Relationship Mapping** using knowledge graphs
- **Contextually-Enhanced Translation** preserving gender and relationship nuances
- **Multi-layered Architecture** with fallback mechanisms for robustness

### How It Works:
1. Text is analyzed to identify named entities and their relationships
2. Coreference resolution links pronouns to their antecedents
3. Gender information is extracted or inferred using multiple methods
4. A knowledge graph is constructed to represent the text's structure
5. Translation incorporates this contextual information for improved accuracy
""")

# API key management
if 'GROQ_API_KEY' not in st.session_state:
    st.session_state['GROQ_API_KEY'] = os.environ.get("GROQ_API_KEY", "")

with st.expander("API Configuration"):
    api_key = st.text_input(
        "Enter Groq API Key", 
        value=st.session_state['GROQ_API_KEY'],
        type="password",
        help="Required for translation and knowledge graph generation"
    )
    if api_key:
        st.session_state['GROQ_API_KEY'] = api_key
        st.success("API Key saved for this session")
    else:
        st.warning("Please enter your Groq API key to use the translation features")

# Quick demo section
st.header("Quick Demo")
demo_text = st.text_area(
    "Enter text to translate",
    value="Kiran is a good student. Sita is his science teacher, and he is Kiran's favorite teacher.",
    height=100
)

col1, col2 = st.columns(2)
source_lang = col1.selectbox("Source Language", ["English"], index=0)
target_lang = col2.selectbox("Target Language", ["Hindi", "Spanish", "French", "German"], index=0)

if st.button("Translate", type="primary", disabled=not st.session_state['GROQ_API_KEY']):
    with st.spinner("Analyzing text and generating knowledge graph..."):
        try:
            # Initialize the translator
            kg_generator = EnhancedKnowledgeGraphGenerator(st.session_state['GROQ_API_KEY'])
            
            # Generate knowledge graph
            graph = kg_generator.create_graph_from_text(demo_text)
            
            # Generate translation prompt
            prompt = kg_generator.generate_translation_prompt(demo_text, graph, target_lang.lower())
            
            # Perform contextual translation
            translation = kg_generator.translate_text(prompt)
            
            # Display results
            st.success("Translation complete!")
            st.subheader("Contextual Translation:")
            st.markdown(f"**{translation}**")
            
            # Store in session state for other pages
            st.session_state['last_graph'] = graph
            st.session_state['last_text'] = demo_text
            st.session_state['last_translation'] = translation
            
            # Suggest exploring other pages
            st.info("Visit the 'Knowledge Graph Explorer' page to visualize the generated knowledge graph!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your API key and try again.")

# Footer
st.markdown("---")
st.markdown("Arthantar - Contextual Translation System")