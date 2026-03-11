#!/usr/bin/env python
"""Update comprehensive_evaluation.py with new Transformer questions."""

import re

file_path = "d:\\Graph_RAG\\experiments\\comprehensive_evaluation.py"

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new questions and references
new_questions = '''self.questions = [
            "What are the main characteristics of the Transformer architecture?",
            "How does Multi-Head Attention relate to Scaled Dot-Product Attention?",
            "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?",
            "Compare the computational complexity per layer of self-attention layers and recurrent layers.",
            "What is the impact of masking in the decoder's self-attention sub-layer?"
        ]'''

new_references = '''self.references = [
            "The Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and using stacked self-attention and point-wise, fully connected layers. [cite: 17, 78]",
            "Multi-Head Attention connects to Scaled Dot-Product Attention by linearly projecting queries, keys, and values h times, and performing the scaled dot-product attention function in parallel on each projected version. [cite: 126, 127]",
            "The Transformer model achieved a new state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU. [cite: 19]",
            "Self-attention layers have a complexity of O(n^2 * d) per layer, while recurrent layers have a complexity of O(n * d^2), making self-attention faster when sequence length n is smaller than representation dimensionality d. [cite: 163, 187, 188, 189]",
            "Masking impacts the decoder by preventing positions from attending to subsequent positions, ensuring that predictions for position i can depend only on the known outputs at positions less than i, preserving the auto-regressive property. [cite: 88, 89]"
        ]'''

# Replace the questions - match the pattern for any questions the __init__ method
questions_pattern = r'self\.questions = \[\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*"\s*\]'
content = re.sub(questions_pattern, new_questions, content, flags=re.DOTALL)

# Replace the references
references_pattern = r'self\.references = \[\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*"\s*\]'
content = re.sub(references_pattern, new_references, content, flags=re.DOTALL)

# Write back the file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Updated {file_path}")
print("\nVerifying updates...")

# Verify by reading back
with open(file_path, 'r', encoding='utf-8') as f:
    updated_content = f.read()
    
if "What are the main characteristics of the Transformer architecture?" in updated_content:
    print("✓ Questions updated successfully!")
    # Find and print the questions section
    import_idx = updated_content.find('self.questions = [')
    print("\nQuestions section:")
    print(updated_content[import_idx:import_idx+300])
else:
    print("✗ Update failed - Transformer question not found!")
