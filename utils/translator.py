import os
from typing import Dict, List, Optional
from fastcoref import FCoref
import networkx as nx
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
import spacy
import json

class EnhancedKnowledgeGraphGenerator:
    def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768"):
        """Initialize the knowledge graph generator with Groq API and coref model"""
        self.llm = ChatGroq(groq_api_key=api_key, model_name=model_name)
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)
        self.coref_model = FCoref()
        # Load spaCy model for backup graph generation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("Warning: spaCy model not loaded. Backup graph generation may be limited.")

    def generate_translation_prompt(self, sample_text: str, knowledge_graph: Dict, target_lang: str = "hindi") -> Dict:
        """Generate a prompt for translation using knowledge graph metadata"""
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
                f"Translate to {target_lang} (use native script) using only the gender and relationship context "
                f"of entities from metadata; reply with only the translated text: {sample_text} "
                f"Meta data: {metadata} "
            )
        }

        return prompt

    def translate_text(self, prompt: Dict) -> str:
        """Translate text using the Groq API"""
        from groq import Groq
        
        # Initialize Groq client
        client = Groq(api_key=self.llm.groq_api_key)
        
        # Create the chat completion
        chat_completion = client.chat.completions.create(
            messages=[prompt],
            model="llama3-8b-8192"
        )
        
        # Return the translated text
        return chat_completion.choices[0].message.content

    def identify_genders_coref(self, text: str) -> Dict[str, str]:
        """Identify genders using coreference resolution"""
        gender_map = {}
        try:
            # Get coreference clusters
            preds = self.coref_model.predict(texts=[text])
            clusters = preds[0].get_clusters()

            # Create sets for male and female indicating pronouns
            male_pronouns = {'he', 'him', 'his'}
            female_pronouns = {'she', 'her', 'hers'}

            for cluster in clusters:
                # Check pronouns in the cluster
                cluster_pronouns = {word.lower() for word in cluster}
                has_male = bool(cluster_pronouns & male_pronouns)
                has_female = bool(cluster_pronouns & female_pronouns)

                # Determine gender based on pronouns
                if has_male and not has_female:
                    gender = 'male'
                elif has_female and not has_male:
                    gender = 'female'
                else:
                    continue  # Skip if no clear gender indicators

                # Assign gender to all non-pronoun entities in the cluster
                for entity in cluster:
                    if (entity.lower() not in male_pronouns and
                        entity.lower() not in female_pronouns and
                        entity[0].isupper()):  # Only proper nouns
                        gender_map[entity] = gender

        except Exception as e:
            print(f"Coreference resolution error: {str(e)}")

        return gender_map

    def identify_gender_llm(self, entity: str, entity_type: str, gender_map: Dict[str, str]) -> str:
        """Use LLM to identify gender when coref fails"""
        # First check if we already have gender from coref
        if entity in gender_map:
            return gender_map[entity]

        if entity_type.lower() != 'person':
            return 'unknown'

        try:
            prompt = f"""
            Analyze the entity name '{entity}' with type '{entity_type}'.
            If it's a person, determine their likely gender based on context.
            Respond with exactly one word: 'male', 'female', or 'unknown'.
            For non-person entities, always respond with 'unknown'.
            """

            response = self.llm.predict(prompt)
            gender = response.lower().strip()

            if gender not in ['male', 'female', 'unknown']:
                return 'unknown'

            return gender

        except Exception as e:
            print(f"LLM gender identification error for {entity}: {str(e)}")
            return 'unknown'

    def create_graph_from_text(self, text: str) -> Dict:
        """Generate a knowledge graph from input text with gender information"""
        try:
            # First pass: Get gender information from coreference resolution
            gender_map = self.identify_genders_coref(text)

            # Generate base graph
            documents = [Document(page_content=text)]
            graph_documents = self.graph_transformer.convert_to_graph_documents(documents)

            if not graph_documents or not hasattr(graph_documents[0], 'nodes'):
                print("Using Basic Graph Generation.")
                return self._create_basic_graph(text, gender_map)

            print("Using LLM-based Graph Generation.")

            # Create NetworkX graph with gender information
            G = nx.DiGraph()

            # Add nodes with gender information
            for node in graph_documents[0].nodes:
                # Use gender from coref map first, then fallback to LLM
                gender = self.identify_gender_llm(node.id, node.type, gender_map)

                G.add_node(node.id,
                          type=node.type,
                          gender=gender,
                          properties=node.properties)

            # Add relationships
            for rel in graph_documents[0].relationships:
                G.add_edge(
                    rel.source.id,
                    rel.target.id,
                    type=rel.type,
                    properties=rel.properties
                )

            return {
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

        except Exception as e:
            print(f"Error in graph generation: {str(e)}")
            return self._create_basic_graph(text, gender_map)

    def _create_basic_graph(self, text: str, gender_map: Dict[str, str]) -> Dict:
        """Create a basic graph structure using spaCy NLP and coreference information"""
        try:
            if self.nlp is None:
                raise ValueError("spaCy model not loaded")
                
            # Process text with spaCy
            doc = self.nlp(text)
            G = nx.DiGraph()
            entities = set()
            entity_mentions = {}  # Track all mentions of each entity

            # First pass: collect named entities and their mentions
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC']:
                    clean_ent = ent.text.strip()
                    entities.add(clean_ent)
                    if clean_ent not in entity_mentions:
                        entity_mentions[clean_ent] = []
                    entity_mentions[clean_ent].append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_
                    })

            # Add nodes with enhanced type detection and gender information
            for entity in entities:
                # Determine entity type from spaCy
                entity_type = 'Entity'  # default
                if entity in entity_mentions:
                    mentions = entity_mentions[entity]
                    if mentions:
                        entity_type = mentions[0]['label']

                # Get gender information, prioritizing the coref map
                gender = gender_map.get(entity, None)
                if gender is None:
                    gender = self.identify_gender_llm(entity, entity_type, gender_map)

                G.add_node(entity, type=entity_type, gender=gender)

            # Extract relationships using dependency parsing
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'pobj']:
                        # Get the head (predicate) and dependent (argument)
                        head = token.head.text
                        dep = token.text

                        # Check if either the head or dependent is in our entities
                        source_entity = None
                        target_entity = None

                        # Find the containing entity for the dependent
                        for ent in entities:
                            if dep in ent or ent in dep:
                                target_entity = ent
                                break

                        # Find any entity that contains the head verb's subject
                        for token2 in sent:
                            if token2.dep_ == 'nsubj' and token2.head == token.head:
                                for ent in entities:
                                    if token2.text in ent or ent in token2.text:
                                        source_entity = ent
                                        break

                        # Add edge if we found both entities
                        if source_entity and target_entity and source_entity != target_entity:
                            relation_type = token.head.text.upper()
                            G.add_edge(source_entity, target_entity,
                                     type=f'REL_{relation_type}')

            # Add relationships based on coreference clusters if not enough found
            if len(G.edges()) < len(entities) - 1:
                entity_list = list(entities)
                for i in range(len(entity_list) - 1):
                    source = entity_list[i]
                    target = entity_list[i + 1]
                    if not G.has_edge(source, target):
                        G.add_edge(source, target, type='RELATED_TO')

            return {
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

        except Exception as e:
            print(f"Error in basic graph generation: {str(e)}")
            # Fallback to extremely basic graph if spaCy fails
            G = nx.DiGraph()
            words = [w for w in text.split() if w[0].isupper()]
            for word in words:
                G.add_node(word, type='Entity',
                          gender=gender_map.get(word, 'unknown'))
            for i in range(len(words) - 1):
                G.add_edge(words[i], words[i + 1], type='RELATED_TO')

            return {
                'nodes': [{
                    'id': n,
                    'type': 'Entity',
                    'gender': G.nodes[n]['gender']
                } for n in G.nodes()],
                'relationships': [{
                    'source': u,
                    'target': v,
                    'type': G.edges[u, v]['type']
                } for u, v in G.edges()]
            }