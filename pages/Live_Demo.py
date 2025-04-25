import streamlit as st
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def check_api_keys():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key not found. Please add it to your .env file.")
        st.code("GROQ_API_KEY=your_api_key_here", language="text")
        return False
    return True

def main():
    st.set_page_config(
        page_title="Live Demo - Arthantar",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Live Demo")
    st.subheader("Try Arthantar with your own text")
    
    # Check if components are initialized
    if 'translator' not in st.session_state or st.session_state.translator is None:
        st.warning("Components not initialized. Please return to the home page to initialize.")
        if st.button("Initialize Components"):
            st.experimental_rerun()
        return
    
    # Input section
    st.markdown("### Enter your text")
    input_text = st.text_area("Text to translate", height=150)
    
    # Language selection
    target_language = st.selectbox(
        "Target Language",
        ["Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Arabic"]
    )
    
    # Translation options
    st.markdown("### Translation Options")
    col1, col2 = st.columns(2)
    
    with col1:
        standard = st.checkbox("Standard Translation", value=True)
    
    with col2:
        enhanced = st.checkbox("Enhanced Translation (with Knowledge Graph)", value=True)
    
    # Process button
    if st.button("Process Text") and input_text:
        if not check_api_keys():
            return
        
        with st.spinner("Processing..."):
            # Generate knowledge graph
            graph = st.session_state.graph_generator.create_graph_from_text(input_text)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Standard Translation", "Enhanced Translation"])
            
            with tab1:
                st.session_state.visualizer.display_graph(graph, input_text)
            
            with tab2:
                if standard:
                    standard_translation = st.session_state.translator.get_standard_translation(
                        input_text, target_language
                    )
                    st.text_area("Standard Translation Result", standard_translation, height=150)
                else:
                    st.info("Standard translation not selected.")
            
            with tab3:
                if enhanced:
                    enhanced_translation = st.session_state.translator.get_enhanced_translation(
                        input_text, graph, target_language
                    )
                    st.text_area("Enhanced Translation Result", enhanced_translation, height=150)
                else:
                    st.info("Enhanced translation not selected.")
    
    # Tips section
    st.markdown("---")
    st.markdown("""
    ### Tips for best results:
    
    1. **Include pronouns**: For the best gender detection, include pronouns (he/she/his/her) in your text.
    2. **Use proper nouns**: Capitalize names of people and places.
    3. **Try complex sentences**: The system works best with sentences that have multiple entities and relationships.
    4. **Compare translations**: See how the enhanced translation differs from the standard one.
    """)

if __name__ == "__main__":
    main()