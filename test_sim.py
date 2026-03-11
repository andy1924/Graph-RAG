import sys
import os
from sentence_transformers import SentenceTransformer, util
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

model = SentenceTransformer("all-MiniLM-L6-v2")

context = "The Transformer architecture is a transduction model that utilizes attention mechanisms, including self-attention and multi-head attention, to process input data."
answer = "The main characteristics of the Transformer architecture are: it is a transduction model that utilizes attention mechanisms, including self-attention and multi-head attention. It features an encoder-decoder structure. It uses embeddings and softmax functions."

answer_sentences = sent_tokenize(answer)
context_sentences = sent_tokenize(context)
print(f"Context sentences: {context_sentences}")
print(f"Answer sentences: {answer_sentences}")

context_embeddings = model.encode(context_sentences, convert_to_tensor=True)
context_full_embedding = model.encode(context, convert_to_tensor=True)

for sentence in answer_sentences:
    if len(sentence.split()) < 5:
        continue
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(sentence_embedding, context_embeddings)
    max_sim = sims.max().item()
    full_sim = util.pytorch_cos_sim(sentence_embedding, context_full_embedding).item()
    print(f"Sentence: {sentence}")
    print(f"  Max Sentence Similarity: {max_sim}")
    print(f"  Full Context Similarity: {full_sim}")


