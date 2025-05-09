import os
from groq import Groq
from typing import Dict, Optional

class EnhancedTranslator:
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        """Initialize the translator with Groq API"""
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=api_key)
    
    def get_standard_translation(self, text: str, target_language: str = "Hindi") -> str:
        """Get a standard translation without context enhancement"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Translate this in {target_language} and send only the {target_language} translated part: {text}",
                    }
                ],
                model=self.model_name,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def get_enhanced_translation(self, text: str, knowledge_graph: Dict, target_language: str = "Hindi") -> str:
        """Get an enhanced translation using knowledge graph context"""
        try:
            # Extract metadata from the knowledge graph
            nodes = knowledge_graph.get('nodes', [])
            relationships = knowledge_graph.get('relationships', [])
            
            # Format nodes
            node_metadata = ', '.join(
                f"{node['id']} (Type: {node['type']}, Gender: {node['gender']})"
                for node in nodes
            )
            
            # Format relationships
            relationship_metadata = ', '.join(
                f"{rel['source']} --[{rel['type']}]--> {rel['target']}"
                for rel in relationships
            )
            
            # Construct metadata string
            metadata = f"Knowledge Graph Structure: Nodes: {node_metadata}. Relationships: {relationship_metadata}."
            
            # Create prompt
            prompt = {
                "role": "user",
                "content": (
                    f"Translate to {target_language} (Devanagari) using only the gender and relationship context of entities from metadata; reply with only the {target_language} text: {text} "
                    f"Meta data: {metadata} "
                )
            }
            
            # Get translation
            chat_completion = self.client.chat.completions.create(
                messages=[prompt],
                model=self.model_name,
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Enhanced translation error: {str(e)}"