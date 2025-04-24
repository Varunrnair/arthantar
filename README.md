🔧 Technical Details - Arthantar
Understanding the Arthantar System
Arthantar is a multi-layered translation enhancement system that ensures gender-awareness and contextual fidelity by combining advanced NLP models, fallback mechanisms, and knowledge graph generation.

🧠 System Architecture
Arthantar employs a three-layered architecture:

Gender Identification Layer

Primary: Uses the FCoref module for coreference-based gender resolution

Backup: Groq LLM for gender prediction when coreference fails

Knowledge Graph Generation Layer

Primary: LLMGraphTransformer with Groq API

Backup: spaCy for named entity and relation extraction

Fallback: Heuristic-based entity extraction using capitalization

Translation Enhancement Layer

Builds gender- and relation-aware prompts for LLM translation

System Flow (Mermaid Diagram):
mermaid
Copy
Edit
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
Technologies Used
FCoref: Coreference resolution

Groq API: LLM backend

spaCy: NLP toolkit

LangChain: Orchestration of LLM workflows

NetworkX: Graph creation

Streamlit: Web interface for interaction

🔁 Coreference Resolution
This module identifies entity clusters in the input text and assigns genders using pronoun resolution. If ambiguous, an LLM is queried for prediction.

Example:
"Kiran is a good student. Sita is his science teacher, and he is Kiran's favorite teacher."
→ Resolves "he" to "Kiran" and infers gender as male.

Fallback: If pronoun-based gender can't be determined, LLM is used.

📊 Knowledge Graph Generation
Entities and relationships are extracted to construct a structured knowledge graph:

Primary: LLM-based graph extraction

Backup: spaCy-based entity/relation parser

Fallback: Rule-based capitalized word extraction

Each node may carry metadata like type and gender. The graph enriches the translation with context.

Applications:

Context-aware translation

Gender consistency

Preservation of semantic relationships