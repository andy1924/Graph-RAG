COMP_EVAL_PATH = r"d:\Graph_RAG\experiments\comprehensive_evaluation.py"

with open(COMP_EVAL_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Google cleanup: remove duplicate references property block
# Line 246 is empty, line 247 is @property, line 266 is empty
# We want to remove lines index 246 to 266 (0-indexed: 245 to 265)
new_lines = []
skip = False
for i, line in enumerate(lines):
    # Remove google duplicate: lines 247-265
    if i >= 246 and i <= 264 and "@property" in line:
        skip = True
    if skip and "class SpaceXCorpus" in line:
        skip = False
    
    # Remove SpaceX trailing garbage
    if "SpaceX Dragon is a reusable spacecraft developed ... [truncated]" in line:
        # We need to backtrack to remove the trailing `,` and items
        # Just skip lines that start with "SpaceX Dragon is a reusable" or "Starlink is SpaceX" or trailing parts
        continue
    if "SpaceX was founded in 2002 by" in line and "truncated" in line:
        continue
    if "The first three Falcon 1 launches" in line and "truncated" in line:
        continue
    if "After stage separation, the Falcon 9 first" in line and "truncated" in line:
        continue
    if "NASA selected SpaceX's Starship as the" in line and "truncated" in line:
        continue
    if "The Raptor is a full-flow staged combustion" in line and "truncated" in line:
        continue
        
    if not skip:
        new_lines.append(line)

# Join back
content = "".join(new_lines)

# Fix SpaceX references list closing
content = content.replace('        ]\n            "SpaceX Dragon is a reusable spacecraft developed ... [truncated]",', '        ]')
# Wait, let's just use regex or exact match to fix the list closure for SpaceXCorpus

with open(COMP_EVAL_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("[+] Cleanup complete")
