from expand_benchmark_defs import *

COMP_EVAL_PATH = r"d:\Graph_RAG\experiments\comprehensive_evaluation.py"

with open(COMP_EVAL_PATH, "r", encoding="utf-8") as f:
    eval_content = f.read()

trigger = "class AttentionPaperCorpus(Corpus):"
footer_trigger = "def run_baseline_experiment("

parts = eval_content.split(trigger)
header = parts[0] + trigger

footer_parts = eval_content.split(footer_trigger)
footer = footer_trigger + footer_parts[1]

# Define fully rigid generators for the 4 corpora for insertion:
def format_list(items):
    lines = ["        return ["]
    for i in items:
        clean = i.replace('"', '\\"')
        lines.append(f'            "{clean}",')
    lines.append("        ]")
    return "\n".join(lines)

attention_class_body = f"""
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

# Stitch together
final_body = attention_body_content = attention_class_body + "\n\n" + tesla_class + "\n\n" + google_class + "\n\n" + spacex_class + "\n\n" + get_all_corp + "\n"

# Verify that stitching doesn't create extra duplicates
rebuilt_content = header + final_body + footer

with open(COMP_EVAL_PATH, "w", encoding="utf-8") as f:
    f.write(rebuilt_content)

print("[+] Re-built comprehensive_evaluation.py successfully!")
