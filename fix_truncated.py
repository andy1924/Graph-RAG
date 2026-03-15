import re
from expand_benchmark_defs import *

COMP_EVAL_PATH = r"d:\Graph_RAG\experiments\comprehensive_evaluation.py"

with open(COMP_EVAL_PATH, "r", encoding="utf-8") as f:
    content = f.read()

def format_list(items):
    lines = ["        return ["]
    for i in items:
        clean = i.replace('"', '\\"')
        lines.append(f'            "{clean}",')
    lines.append("        ]")
    return "\n".join(lines)

# Create accurate replacement for Google's references
google_refs = format_list(GOOGLE_REFERENCES)
google_sect_regex = r"(class GoogleCorpus\(Corpus\):.*?def references\(self\) -> List\[str\]:\n).*?(\[.*?\])\n"

content = re.sub(
    r"(class GoogleCorpus\(Corpus\):.*?def references\(self\) -> List\[str\]:\n\s+return\s+\[).*?(\])",
    f"\\1\n{',\n'.join([f'            "{r.replace('"', '\\"')}"' for r in GOOGLE_REFERENCES])}\n        \\2",
    content,
    flags=re.DOTALL
)

content = re.sub(
    r"(class SpaceXCorpus\(Corpus\):.*?def references\(self\) -> List\[str\]:\n\s+return\s+\[).*?(\])",
    f"\\1\n{',\n'.join([f'            "{r.replace('"', '\\"')}"' for r in SPACEX_REFERENCES])}\n        \\2",
    content,
    flags=re.DOTALL
)

with open(COMP_EVAL_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("[+] Fixed truncated references for Google and SpaceX")
