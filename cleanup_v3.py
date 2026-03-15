import re
COMP_EVAL_PATH = r"d:\Graph_RAG\experiments\comprehensive_evaluation.py"

with open(COMP_EVAL_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Index mapping based on view_file output:
# Line 247: @property
# Line 266: ]
# We want to remove lines 247-266 inclusive (1-indexed)
# Wait, let's just find where duplicate stars and delete it.

# Look for: '@property' and then 'def references(self) -> List[str]:'
# Since we know there are only TWO of these in GoogleCorpus (or we can just match it).
# A safer way is to find the FIRST references list and keep it, and find the SECOND and DELETE it.

output_lines = []
google_corpus_started = False
references_count = 0

i = 0
while i < len(lines):
    line = lines[i]
    if "class GoogleCorpus" in line:
        google_corpus_started = True
    if "class SpaceXCorpus" in line:
        google_corpus_started = False
        
    if google_corpus_started and "@property" in line and i+1 < len(lines) and "def references(self)" in lines[i+1]:
        references_count += 1
        if references_count > 1:
            # This is the duplicate! Skip until we find the closing bracket ] and next class or function
            # Skip this line and everything in the list
            while i < len(lines) and "]" not in lines[i]:
                i += 1
            i += 1 # skip the closing bracket ]
            if i < len(lines) and lines[i].strip() == "":
                i += 1 # skip spacing
            continue
            
    # SpaceX cleanup: trailing items from line 314
    if "class SpaceXCorpus" in line:
        # Just let it pass, but when we reach references, we will reconstruct it properly
        pass
        
    output_lines.append(lines[i])
    i += 1

# Reconstruct SpaceXReferences completely just to be safe
content = "".join(output_lines)

# Now fix SpaceX references specifically by replacing the whole block with the expansion defs
from expand_benchmark_defs import SPACEX_REFERENCES

spacex_refs_block = "    @property\n    def references(self) -> List[str]:\n        return [\n"
for r in SPACEX_REFERENCES:
    spacex_refs_block += f'            "{r.replace('"', '\\"')}",\n'
spacex_refs_block += "        ]\n"

content = re.sub(
    r"def references\(self\) -> List\[str\]:\n\s+return\s+\[.*?\]",
    spacex_refs_block.replace("\n", "__NEWLINE__"),
    content,
    flags=re.DOTALL
)
# Wait, that would match ALL of them. We need to match only SpaceX corpus.

import re
content = re.sub(
    r"class SpaceXCorpus\(Corpus\):.*?def references\(self\) -> List\[str\]:\n\s+return\s+\[.*?\]",
    lambda m: m.group(0).split("@property")[0] + spacex_refs_block,
    content,
    flags=re.DOTALL
)

with open(COMP_EVAL_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("[+] Cleaned up file")
