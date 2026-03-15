import re
from expand_benchmark_defs import *

COMP_EVAL_PATH = r"d:\Graph_RAG\experiments\comprehensive_evaluation.py"
DATA_RET_PATH = r"d:\Graph_RAG\src\graphrag\utils\data_retriever.py"

# =====================================================================
# 1. Update comprehensive_evaluation.py
# =====================================================================
with open(COMP_EVAL_PATH, "r", encoding="utf-8") as f:
    eval_content = f.read()

# Helper to generate is-list string
def format_list(items):
    lines = ["        return ["]
    for i in items:
        clean = i.replace('"', '\\"')
        lines.append(f'            "{clean}",')
    lines.append("        ]")
    return "\n".join(lines)

# Define full class replacements
attention_class = f"""class AttentionPaperCorpus(Corpus):
    \"\"\"15 QA pairs about the 'Attention Is All You Need' paper.\"\"\"

    @property
    def corpus_id(self) -> str:
        return "attention_paper"

    @property
    def questions(self) -> List[str]:
{format_list(ATTENTION_QUESTIONS)}

    @property
    def references(self) -> List[str]:
        raw = [
{",\n".join([f'            "{r.replace('"', '\\"')}"' for r in ATTENTION_REFERENCES])}
        ]
        return [_strip_cite_markers(r) for r in raw]

    @property
    def relevant_items(self) -> List[List[str]]:
        # Retrieve relevant items from actual graph data
        print("\\n" + "=" * 60)
        print("Loading relevant items from graph data (Attention Paper)...")
        print("=" * 60)
        items = get_relevant_items_mapping(
            self.questions,
            question_keywords=QUESTION_KEYWORDS_MAPPING,
        )
        print("=" * 60 + "\\n")
        return items
"""

tesla_class = f"""class TeslaCorpus(Corpus):
    \"\"\"15 QA pairs about Tesla, derived from data/raw/Tesla.txt.\"\"\"

    @property
    def corpus_id(self) -> str:
        return "tesla"

    @property
    def questions(self) -> List[str]:
{format_list(TESLA_QUESTIONS)}

    @property
    def references(self) -> List[str]:
{format_list(TESLA_REFERENCES)}
"""

google_class = f"""class GoogleCorpus(Corpus):
    \"\"\"15 QA pairs about Google, derived from data/raw/Google.txt.\"\"\"

    @property
    def corpus_id(self) -> str:
        return "google"

    @property
    def questions(self) -> List[str]:
{format_list(GOOGLE_QUESTIONS)}

    @property
    def references(self) -> List[str]:
{format_list(GOOGLE_REFERENCES)}
"""

spacex_class = f"""class SpaceXCorpus(Corpus):
    \"\"\"15 QA pairs about SpaceX, derived from data/raw/SpaceX.txt.\"\"\"

    @property
    def corpus_id(self) -> str:
        return "spacex"

    @property
    def questions(self) -> List[str]:
{format_list(SPACEX_QUESTIONS)}

    @property
    def references(self) -> List[str]:
{format_list(SPACEX_REFERENCES)}
"""

# Replace Attention
eval_content = re.sub(
    r"class AttentionPaperCorpus\(Corpus\):.*?def relevant_items\(self\).*?return items\n",
    attention_class,
    eval_content,
    flags=re.DOTALL
)

# Replace Tesla
eval_content = re.sub(
    r"class TeslaCorpus\(Corpus\):.*?\]\n",
    tesla_class,
    eval_content,
    flags=re.DOTALL
)

# Replace Google
eval_content = re.sub(
    r"class GoogleCorpus\(Corpus\):.*?\]\n",
    google_class,
    eval_content,
    flags=re.DOTALL
)

# Insert SpaceX before get_all_corpora if not present
if "class SpaceXCorpus" not in eval_content:
    eval_content = eval_content.replace(
        "def get_all_corpora()",
        spacex_class + "\n\ndef get_all_corpora()"
    )

# Update get_all_corpora
eval_content = re.sub(
    r"def get_all_corpora\(\) -> List\[Corpus\]:.*?return \[.*?\]",
    "def get_all_corpora() -> List[Corpus]:\n    \"\"\"Return all available evaluation corpora.\"\"\"\n    return [\n        AttentionPaperCorpus(),\n        TeslaCorpus(),\n        GoogleCorpus(),\n        SpaceXCorpus(),\n    ]",
    eval_content,
    flags=re.DOTALL
)

with open(COMP_EVAL_PATH, "w", encoding="utf-8") as f:
    f.write(eval_content)
print("[+] Updated comprehensive_evaluation.py")

# =====================================================================
# 2. Update data_retriever.py
# =====================================================================
with open(DATA_RET_PATH, "r", encoding="utf-8") as f:
    ret_content = f.read()

# Replace GROUND_TRUTH_RELEVANT_ITEMS
gt_replacement = """GROUND_TRUTH_RELEVANT_ITEMS: Dict[str, List[str]] = {
    "What are the main characteristics of the Transformer architecture?": [
        "Transformer", "Attention Mechanisms", "Self-Attention", "Self-Attention Mechanism",
        "Encoder", "Decoder", "Encoder-Decoder Structure", "Multi-Head Attention", "Attention Is All You Need"
    ],
    "How does Multi-Head Attention relate to Scaled Dot-Product Attention?": [
        "Multi-Head Attention", "Scaled Dot-Product Attention", "Attention Head", "Attention Function",
        "Multi-Head Self-Attention Mechanism", "Softmax", "Queries", "Keys", "Values"
    ],
    "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?": [
        "Transformer", "Transformer (Big)", "Transformer (Base Model)", "Wmt 2014 English-To-German Translation Task",
        "Wmt 2014 English-German Dataset", "Attention Is All You Need"
    ],
    "Compare the computational complexity per layer of self-attention layers and recurrent layers.": [
        "Self-Attention", "Self-Attention Mechanism", "Recurrent", "Recurrent Layers",
        "Recurrent Neural Networks", "Recurrent Language Models"
    ],
    "What is the impact of masking in the decoder's self-attention sub-layer?": [
        "Masking", "Decoder", "Self-Attention", "Self-Attention Mechanism",
        "Encoder-Decoder Attention", "Multi-Head Self-Attention Mechanism"
    ],
    "What role do Positional Encodings play in the Transformer and why are they necessary given the absence of recurrence?": [
        "Positional Encoding", "Transformer", "Self-Attention", "Recurrent Neural Networks"
    ],
    "How do residual connections and layer normalization contribute to training stability in the Transformer?": [
        "Residual Connection", "Layer Normalization", "Transformer", "Encoder", "Decoder", "Multi-Head Attention"
    ],
    "Who authored the Transformer paper and what institutions were they affiliated with?": [
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones",
        "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin", "Attention Is All You Need",
        "Google Brain", "University of Toronto"
    ],
    "How does the Transformer Big model compare to ConvS2S and GNMT ensembles on the WMT 2014 English-to-German task?": [
        "Transformer (Big)", "ConvS2S", "Convs2S [9]", "Gnmt + Rl Ensemble [38]", "Wmt 2014 English-To-German Translation Task"
    ],
    "What are the dimensions and structure of the Feed-Forward sublayer in the Transformer base model?": [
        "Feed-Forward Network", "Transformer (Base Model)"
    ],
    "What dropout rate and where is dropout applied in the Transformer training procedure?": [
        "Dropout", "Transformer", "Transformer (Base Model)"
    ],
    "What vocabulary size and encoding scheme is used for WMT 2014 English-to-German versus English-to-French?": [
        "Byte-Pair Encoding", "Wmt 2014 English-To-German Translation Task"
    ],
    "What are the number of attention heads and key/value dimensions in the Transformer base versus big model?": [
        "Transformer (Base Model)", "Transformer (Big)", "Multi-Head Attention"
    ],
    "What Adam optimizer hyperparameters were used to train the Transformer?": [
        "Adam Optimizer", "Transformer"
    ],
    "What results did the Transformer achieve on English constituency parsing, and how does it generalize?": [
        "English Constituency Parsing", "Wall Street Journal Portion Of The Penn Treebank", "Transformer"
    ],
    # Tesla
    "How did Tesla's acquisitions of Maxwell Technologies and Hibar Systems relate to its battery technology strategy?": [
        "Maxwell Technologies", "Hibar Systems", "Battery", "Acquisition"
    ],
    "What is the relationship between Tesla Energy, SolarCity, and Tesla's energy product portfolio?": [
        "Tesla Energy", "SolarCity", "Energy", "Acquisition"
    ],
    "How has Tesla's North American Charging Standard affected its relationship with competitors in the EV industry?": [
        "North American Charging Standard", "NACS", "Competitor", "Standard"
    ],
    "Which Gigafactories does Tesla operate and what does each primarily produce?": [
        "Gigafactory", "Grid", "Factory"
    ],
    "Who are Tesla's primary battery cell suppliers and what cell formats does each supply?": [
        "Panasonic", "LG Energy Solution", "CATL", "Battery"
    ],
    "What NHTSA investigations has Tesla's Autopilot and Full Self-Driving been subject to?": [
        "NHTSA", "Investigation", "Autopilot", "Full Self-Driving"
    ],
    "What major legal disputes has Tesla faced involving its founders and employees?": [
        "Legal dispute", "Lawsuit", "Founder", "Employee"
    ],
    "What is Tesla's Optimus robot and what is its development status?": [
        "Optimus", "Robot", "Development"
    ],
    "How has Tesla's annual revenue trended from 2017 to 2024?": [
        "Revenue", "Financial"
    ],
    "What is Tesla's insurance business and in which states does it operate?": [
        "Insurance", "State"
    ],
    # Google
    "How did Google's acquisitions of YouTube and DoubleClick shape its advertising and video dominance?": [
        "YouTube", "DoubleClick", "Advertising", "Acquisition"
    ],
    "What is the relationship between DeepMind Technologies, AlphaGo, and Google's generative AI strategy?": [
        "DeepMind Technologies", "AlphaGo", "Generative AI", "AI strategy"
    ],
    "What antitrust fines has Google received from European regulators and on what grounds?": [
        "Antitrust", "Fine", "European regulators", "Grounds"
    ],
    "How did Google's '20% Innovation Time Off' policy lead to the creation of Gmail, Google News, and AdSense?": [
        "20% Innovation Time Off", "Gmail", "Google News", "AdSense"
    ],
    "What was Eric Schmidt's role at Google and how did the leadership transition to Sundar Pichai occur?": [
        "Eric Schmidt", "Sundar Pichai", "Leadership transition", "CEO"
    ],
    "What was Project Nightingale and why did it draw criticism?": [
        "Project Nightingale", "Criticism", "Patient privacy"
    ],
    "What submarine cable infrastructure does Google operate globally?": [
        "Submarine cable", "Infrastructure"
    ],
    "How does Google's tax strategy using Ireland, the Netherlands, and Bermuda minimize its tax liability?": [
        "Tax strategy", "Ireland", "Netherlands", "Bermuda"
    ],
    "What were the causes and outcomes of the 2018 Google Walkout?": [
        "2018 Google Walkout", "Walkout", "Causes", "Outcomes"
    ],
    "What was Google's Motorola Mobility acquisition and what happened to it?": [
        "Motorola Mobility", "Acquisition", "Device business"
    ]
}"""

# Replace QUESTION_KEYWORDS_MAPPING
kw_replacement = """QUESTION_KEYWORDS_MAPPING = {
    # Attention Paper
    "What are the main characteristics of the Transformer architecture?": ["Transformer", "architecture", "attention", "self-attention"],
    "How does Multi-Head Attention relate to Scaled Dot-Product Attention?": ["Multi-Head Attention", "Scaled Dot-Product", "attention", "projection"],
    "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?": ["Transformer", "WMT 2014", "translation", "BLEU", "performance"],
    "Compare the computational complexity per layer of self-attention layers and recurrent layers.": ["self-attention", "recurrent", "complexity", "computational", "O(n)"],
    "What is the impact of masking in the decoder's self-attention sub-layer?": ["masking", "decoder", "self-attention", "positions", "autoregressive"],
    "What role do Positional Encodings play in the Transformer and why are they necessary given the absence of recurrence?": ["Positional Encodings", "Transformer", "recurrence", "encoding"],
    "How do residual connections and layer normalization contribute to training stability in the Transformer?": ["residual connections", "layer normalization", "stability", "training"],
    "Who authored the Transformer paper and what institutions were they affiliated with?": ["authored", "Transformer paper", "institutions", "affiliated"],
    "How does the Transformer Big model compare to ConvS2S and GNMT ensembles on the WMT 2014 English-to-German task?": ["Transformer Big", "ConvS2S", "GNMT", "WMT 2014"],
    "What are the dimensions and structure of the Feed-Forward sublayer in the Transformer base model?": ["dimensions", "structure", "Feed-Forward", "base model"],
    "What dropout rate and where is dropout applied in the Transformer training procedure?": ["dropout", "rate", "applied", "training procedure"],
    "What vocabulary size and encoding scheme is used for WMT 2014 English-to-German versus English-to-French?": ["vocabulary size", "encoding scheme", "WMT 2014", "German", "French"],
    "What are the number of attention heads and key/value dimensions in the Transformer base versus big model?": ["attention heads", "key/value", "dimensions", "base", "big"],
    "What Adam optimizer hyperparameters were used to train the Transformer?": ["Adam optimizer", "hyperparameters", "train", "Transformer"],
    "What results did the Transformer achieve on English constituency parsing, and how does it generalize?": ["results", "Transformer", "English constituency parsing", "generalize"],
    # Tesla
    "What are Tesla's main product lines as of November 2024?": ["Tesla", "product lines", "vehicle", "energy"],
    "What was Tesla's total revenue in 2024?": ["revenue", "2024", "financial"],
    "What is the current status and timeline of Tesla's Full Self-Driving technology?": ["Full Self-Driving", "status", "timeline", "autonomous"],
    "Who leads Tesla and what is the company's leadership structure?": ["leads Tesla", "leadership structure", "management", "Elon Musk"],
    "Who are Tesla's main competitors and partners in the EV market?": ["competitors", "partners", "EV market", "Tesla"],
    # SpaceX (Keywords only fallback)
    "What is SpaceX's Falcon 9 and what makes it significant in launch history?": ["Falcon 9", "SpaceX", "launch history", "reusable"],
    "What is the Starship vehicle and what is its intended purpose?": ["Starship", "intended purpose", "vehicle", "SpaceX"],
    "How did SpaceX's Dragon spacecraft contribute to International Space Station resupply?": ["Dragon", "International Space Station", "resupply", "SpaceX"],
    "What is Starlink and how does it relate to SpaceX's business model?": ["Starlink", "SpaceX", "business model", "satellite internet"],
    "Who founded SpaceX and what was the original motivation for starting the company?": ["founded SpaceX", "original motivation", "starting company", "Elon Musk"],
    "What were the first three Falcon 1 launches and why were they significant?": ["Falcon 1 launches", "significant", "SpaceX"],
    "How does SpaceX's first-stage booster recovery and reuse work?": ["booster recovery", "reuse", "first-stage", "SpaceX"],
    "What is the relationship between SpaceX and NASA's Artemis Moon program?": ["SpaceX", "NASA", "Artemis Moon", "relationship"],
    "What is the Raptor engine and how does it differ from Merlin?": ["Raptor engine", "Merlin", "differ", "SpaceX"],
    "What is Falcon Heavy and what has it been used for?": ["Falcon Heavy", "used for", "SpaceX"],
    "How does SpaceX's Starlink compete with traditional satellite internet providers?": ["Starlink", "compete", "traditional", "satellite internet"],
    "What regulatory and legal challenges has SpaceX faced for Starship launches?": ["regulatory", "legal challenges", "Starship launches", "SpaceX"],
    "What is SpaceX's Crew Dragon and how did it end US dependence on Russia for ISS access?": ["Crew Dragon", "end dependence", "Russia", "ISS access"],
    "What is SpaceX's long-term vision for Mars colonization?": ["long-term vision", "Mars colonization", "SpaceX"],
    "How does SpaceX generate revenue across its different business lines?": ["generate revenue", "business lines", "SpaceX"]
}"""

ret_content = re.sub(
    r"GROUND_TRUTH_RELEVANT_ITEMS: Dict\[str, List\[str\]\] = \{.*?\}\n",
    gt_replacement + "\n",
    ret_content,
    flags=re.DOTALL
)

ret_content = re.sub(
    r"QUESTION_KEYWORDS_MAPPING = \{.*?\}\n",
    kw_replacement + "\n",
    ret_content,
    flags=re.DOTALL
)

with open(DATA_RET_PATH, "w", encoding="utf-8") as f:
    f.write(ret_content)
print("[+] Updated data_retriever.py")
print("[+] Done")
