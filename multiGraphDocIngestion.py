import os
import json
import base64
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

import logging

# Silence the specific pdfminer warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def encode_image(image_path):
    """Encodes image for MLLM vision processing."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def summarize_multimodal_element(llm, content, is_image=False):
    """Summarizes visual/tabular data to prevent information decay."""
    if is_image:
        # Vision-based summary for the Element Subgraph
        msg = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Describe this image/chart for a knowledge graph. Focus on relationships."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}
            ])
        ])
        return msg.content
    else:
        # Table structural summary
        return llm.invoke(f"Summarize this table data for graph nodes: {content}").content


def run_ingestion(input_path):
    """Primary ingestion logic for multimodal PDF partitioning."""
    llm = ChatOpenAI(model='gpt-4o')  # Vision capable model

    # PDF Partitioning (Addressing the Multimodal Gap)
    elements = partition_pdf(
        filename=input_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        extract_image_block_output_dir="data/extracted_images"
    )

    data_payload = []
    for i, el in enumerate(elements):
        el_type = str(type(el))
        metadata = el.metadata.to_dict()
        node = {"id": f"node_{i}", "metadata": metadata}

        if "Table" in el_type:
            # Preserve structural hierarchy for tables
            node["type"] = "Table"
            node["content"] = summarize_multimodal_element(llm, metadata.get("text_as_html", str(el)))
        elif "Image" in el_type:
            # Handle visual cues via MLLM
            img_path = metadata.get("image_path")
            if img_path and os.path.exists(img_path):
                node["type"] = "Image"
                node["content"] = summarize_multimodal_element(llm, encode_image(img_path), is_image=True)
            else:
                continue
        else:
            # Standard lexical text
            node["type"] = "Text"
            node["content"] = str(el)

        data_payload.append(node)
    return data_payload


def main():
    # Setup paths
    input_file = r"data\multiModalPDF\Attention_Is_All_You_Need_RP.pdf"
    output_file = "data/preprocessed/graph_data.json"

    # Cost and existence check
    if os.path.exists(output_file):
        print("!! WARNING: Preprocessed data already exists.")
        if input("This process is expensive. Overwrite? (y/n): ").lower() != 'y':
            return

    print("--- Starting Multimodal Ingestion ---")
    try:
        nodes = run_ingestion(input_file)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=4)
        print(f"--- Successfully preprocessed {len(nodes)} nodes ---")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()