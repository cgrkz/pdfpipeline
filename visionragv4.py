!sudo apt-get install -y -q poppler-utils
!pip install -U -q byaldi pdf2image qwen-vl-utils transformers bitsandbytes peft matplotlib gradio

import os
import shutil
import gradio as gr
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import torch
import logging

# --- 1. Setup, Configuration & New Pixel Class ---

# Configure logging to provide detailed terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define directories for data and index storage
DATA_DIR = "data"
INDEX_DIR = "image_index"
os.makedirs(DATA_DIR, exist_ok=True)


class PixelConfigurations:
    """Comprehensive pixel configuration options for Qwen2.5-VL"""
    # Configuration dictionaries
    BALANCED = {
        "min_pixels": 256 * 28 * 28,  # 200,704 pixels
        "max_pixels": 768 * 28 * 28,  # 601,344 pixels
        "description": "✅ Balanced (Default): Good speed and high accuracy for general use."
    }
    FAST = {
        "min_pixels": 128 * 28 * 28,  # 100,352 pixels
        "max_pixels": 256 * 28 * 28,  # 200,704 pixels
        "description": "⚡ Fast: Prioritizes speed for quick scans. Lower detail."
    }
    HIGH_QUALITY = {
        "min_pixels": 512 * 28 * 28,  # 401,408 pixels
        "max_pixels": 1024 * 28 * 28, # 802,816 pixels
        "description": "🖼️ High Quality: For detailed charts and complex scientific papers."
    }
    ULTRA_HIGH = {
        "min_pixels": 512 * 28 * 28,  # 401,408 pixels
        "max_pixels": 2048 * 28 * 28, # 1,605,632 pixels
        "description": "🔬 Ultra High: Maximum detail for archival/research-grade analysis."
    }

    @classmethod
    def get_choices(cls):
        """Returns a list of tuples for Gradio dropdown, e.g., [('Balanced', 'BALANCED')]"""
        return [(v['description'], k) for k, v in cls.__dict__.items() if isinstance(v, dict)]

    @classmethod
    def get_config(cls, name):
        """Returns the configuration dictionary for a given name."""
        return getattr(cls, name, cls.BALANCED)


# --- 2. Model Loading ---

def load_models():
    """
    Loads the main, heavy models once.
    This includes the retrieval model and the vision-language model.
    The processor is loaded separately to allow for dynamic configuration changes.
    """
    print("--- Starting to load main models (this may take a moment) ---")
    logging.info("Starting to load main models...")

    # Load the document retrieval model
    print("Loading retrieval model: vidore/colqwen2-v1.0...")
    logging.info("Loading retrieval model: vidore/colqwen2-v1.0")
    docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")
    print("✅ Retrieval model loaded.")
    logging.info("Retrieval model loaded successfully.")

    # Configure quantization for the vision-language model (for memory efficiency)
    print("Configuring BitsAndBytes for Qwen2.5-VL-7B-Instruct...")
    logging.info("Configuring BitsAndBytes for Qwen2.5-VL-7B-Instruct...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the vision-language model
    print("Loading VL model: Qwen/Qwen2.5-VL-7B-Instruct...")
    logging.info("Loading VL model: Qwen/Qwen2.5-VL-7B-Instruct")
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    vl_model.eval()
    print("✅ Vision-Language model loaded.")
    logging.info("VL model loaded successfully.")

    return docs_retrieval_model, vl_model

def load_processor(config_name="BALANCED"):
    """
    Loads just the vision-language model processor with a specific pixel configuration.
    """
    config = PixelConfigurations.get_config(config_name)
    min_pixels = config["min_pixels"]
    max_pixels = config["max_pixels"]

    print(f"--- Loading processor with configuration: {config_name} ---")
    print(f"Min Pixels: {min_pixels}, Max Pixels: {max_pixels}")
    logging.info(f"Loading VL processor with config: {config_name} ({min_pixels}-{max_pixels}px)")

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    print("✅ Processor loaded.")
    logging.info("VL model processor loaded successfully.")
    return processor


# Load models globally; processor is also loaded with a default setting.
docs_retrieval_model, vl_model = load_models()
vl_model_processor = load_processor("BALANCED") # Load default processor

# Global dictionary to store processed images from PDFs
all_images = {}


# --- 3. Core RAG Pipeline Functions ---

def process_uploaded_pdfs(files):
    """
    Handles uploaded PDF files by saving them, converting them to images,
    and building a searchable index.
    """
    if not files:
        return "Please upload at least one PDF file to begin.", None, []

    print(f"\n--- Received {len(files)} PDF files for processing ---")
    logging.info(f"Received {len(files)} PDF files for processing.")

    # Clear previous data
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    print(f"Cleared and recreated data directory: {DATA_DIR}")

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    print(f"Cleared and recreated index directory: {INDEX_DIR}")


    global all_images
    all_images.clear()

    # Save uploaded files
    for file in files:
        shutil.copy(file.name, os.path.join(DATA_DIR, os.path.basename(file.name)))
    print(f"PDFs copied to {DATA_DIR}/")
    logging.info(f"PDFs copied to {DATA_DIR}/")

    # Convert PDFs to images
    print("Converting PDFs to images...")
    logging.info("Converting PDFs to images...")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        try:
            images = convert_from_path(pdf_path)
            all_images[doc_id] = images
            print(f"Converted {pdf_file} into {len(images)} images.")
            logging.info(f"Converted {pdf_file} into {len(images)} images.")
        except Exception as e:
            print(f"ERROR: Failed to convert {pdf_file}: {e}")
            logging.error(f"Failed to convert {pdf_file}: {e}")
            return f"Error processing {pdf_file}. Please check if it's a valid PDF.", None, []

    # Build the retrieval index
    print("Building retrieval index from images...")
    logging.info("Building retrieval index from images...")
    docs_retrieval_model.index(
        input_path=DATA_DIR,
        index_name=INDEX_DIR,
        store_collection_with_index=False,
        overwrite=True,
    )
    print("✅ Index built successfully.")
    logging.info("Index built successfully.")

    return f"✅ Successfully processed and indexed {len(files)} PDF(s). You can now ask questions.", None, []

def get_grouped_images(results):
    """
    Retrieves the actual image objects based on the search results.
    """
    grouped_images = []
    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        if doc_id in all_images and 0 < page_num <= len(all_images[doc_id]):
            # page_num is 1-indexed, list is 0-indexed
            grouped_images.append(all_images[doc_id][page_num - 1])
    return grouped_images

def answer_question(chat_history, text_query):
    """
    The main chat function that takes a user query, performs RAG,
    and returns the answer and retrieved images.
    """
    if not all_images:
        chat_history.append((text_query, "Error: No PDFs have been processed. Please upload PDFs first."))
        return chat_history, []

    print(f"\n--- Received query: '{text_query}' ---")
    logging.info(f"Received query: '{text_query}'")
    top_k = 3
    max_new_tokens = 1024

    # 1. Search for relevant images
    print("Searching for relevant images...")
    logging.info("Searching for relevant images...")
    results = docs_retrieval_model.search(text_query, k=top_k)
    retrieved_images = get_grouped_images(results)
    print(f"Retrieved {len(retrieved_images)} images for the query.")
    logging.info(f"Retrieved {len(retrieved_images)} images for the query.")

    if not retrieved_images:
        chat_history.append((text_query, "Could not find relevant images in the documents for your query."))
        return chat_history, []

    # 2. Prepare inputs for the VL model
    print("Preparing inputs for the vision-language model...")
    logging.info("Preparing inputs for the vision-language model...")
    chat_template = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in retrieved_images]
            + [{"type": "text", "text": text_query}],
        }
    ]
    
    # Use the global, potentially updated, processor
    global vl_model_processor
    text = vl_model_processor.apply_chat_template(
        chat_template, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 3. Generate the answer
    print("Generating answer...")
    logging.info("Generating answer...")
    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("✅ Answer generated.")
    logging.info("Answer generated successfully.")

    chat_history.append((text_query, output_text))

    return chat_history, retrieved_images


# --- 4. Gradio User Interface ---

def handle_config_change(config_name):
    """
    Updates the global processor when the user changes the configuration in the UI.
    """
    global vl_model_processor
    vl_model_processor = load_processor(config_name)
    print(f"✅ Configuration updated to {config_name}")
    return f"Configuration updated to: {config_name}"


with gr.Blocks(theme=gr.themes.Soft(), title="Multimodal RAG with Qwen-VL") as demo:
    gr.Markdown("# Multimodal RAG with Qwen-VL 💬🖼️")
    gr.Markdown(
        "Upload PDFs, which will be processed and indexed. Then, ask questions about their content. "
        "The model will retrieve relevant images from the PDFs to answer your questions."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Upload & Process PDFs")
            pdf_upload = gr.File(
                label="Upload PDF Files", file_count="multiple", file_types=[".pdf"]
            )
            process_button = gr.Button("Process PDFs", variant="primary")
            upload_status = gr.Textbox(label="Processing Status", interactive=False, lines=2)

            # --- NEW: Configuration Dropdown ---
            gr.Markdown("## Settings")
            config_dropdown = gr.Dropdown(
                label="Image Quality/Speed Configuration",
                choices=PixelConfigurations.get_choices(),
                value="BALANCED",
                interactive=True
            )
            apply_config_button = gr.Button("Apply Configuration")
            config_status = gr.Textbox(label="Configuration Status", interactive=False)
            # --- END NEW ---


        with gr.Column(scale=2):
            gr.Markdown("## 2. Chat with your Documents")
            chatbot = gr.Chatbot(label="Chat History", height=500, bubble_full_width=False)
            with gr.Row():
                query_box = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g., Based on the chart, what were the main findings?",
                    scale=4,
                )
                submit_button = gr.Button("Submit", variant="primary", scale=1)
            gr.Markdown("### Retrieved Images for Last Query")
            retrieved_images_display = gr.Gallery(
                label="Retrieved Images",
                show_label=False,
                elem_id="gallery",
                columns=3,
                height="auto",
            )

    # --- Event Handlers ---
    process_button.click(
        fn=process_uploaded_pdfs,
        inputs=[pdf_upload],
        outputs=[upload_status, chatbot, retrieved_images_display],
    )

    submit_button.click(
        fn=answer_question,
        inputs=[chatbot, query_box],
        outputs=[chatbot, retrieved_images_display],
    ).then(lambda: gr.update(value=""), None, [query_box], queue=False)

    query_box.submit(
        fn=answer_question,
        inputs=[chatbot, query_box],
        outputs=[chatbot, retrieved_images_display],
    ).then(lambda: gr.update(value=""), None, [query_box], queue=False)

    # --- NEW: Handler for configuration change ---
    apply_config_button.click(
        fn=handle_config_change,
        inputs=[config_dropdown],
        outputs=[config_status]
    )


if __name__ == "__main__":
    print("--- Starting Gradio application ---")
    logging.info("Starting Gradio application...")
    demo.launch(debug=True, share=True)
