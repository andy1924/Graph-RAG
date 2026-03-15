from expand_benchmark_defs import *

COMP_EVAL_PATH = r"d:\Graph_RAG\experiments\comprehensive_evaluation.py"

with open(COMP_EVAL_PATH, "r", encoding="utf-8") as f:
    orig = f.read()

# We need to preserve everything BEFORE AttentionPaperCorpus
# AND everything AFTER get_all_corpora

# Find split index for start of corpora
trigger = "class AttentionPaperCorpus(Corpus):"
if trigger not in orig:
    # Fail safe: find earliest corpus
    trigger_index = orig.find("class Corpus")
    # Actually, we can just split on first corpus class that is known.
    pass

parts = orig.split(trigger)
header = parts[0] + trigger

footer_trigger = "def run_baseline_experiment("
footer = orig.split(footer_trigger)[1]

def format_list(items):
    lines = ["        return ["]
    for i in items:
        clean = i.replace('"', '\\"')
        lines.append(f'            "{clean}",')
    lines.append("        ]")
    return "\n".join(lines)

# Re-generate Attention class body to keep relevant_items property intact
attention_body = f"""
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

tesla_class = f"""
class TeslaCorpus(Corpus):
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

google_class = f"""
class GoogleCorpus(Corpus):
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

spacex_class = f"""
class SpaceXCorpus(Corpus):
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

get_all_corp = """
def get_all_corpora() -> List[Corpus]:
    \"\"\"Return all available evaluation corpora.\"\"\"
    return [
        AttentionPaperCorpus(),
        TeslaCorpus(),
        GoogleCorpus(),
        SpaceXCorpus(),
    ]

"""

# Stitch back
body_complete = attention_body + "\n" + tesla_class + "\n" + google_class + "\n" + spacex_class + "\n" + get_all_corp

result = header + body_complete + footer_trigger + footer

with open(COMP_EVAL_PATH, "w", encoding="utf-8") as f:
    f.write(result)

print("[+] Overwritten comprehensive_evaluation.py successfully")
