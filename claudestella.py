#!/usr/bin/env python3

"""
FIXED Enhanced Multi-PDF RAG Pipeline with LlamaIndex and Stella Embed
==========================================================================

Version 9.3 (FIXED RAG ISSUES):
- FIXED: Score interpretation for FAISS IndexFlatL2 (lower scores = more similar)
- FIXED: Document filtering logic that was removing relevant documents
- FIXED: Improved chunking strategy for better context preservation
- FIXED: Better score normalization and thresholds
- FIXED: Enhanced debug output for better troubleshooting
- FIXED: Improved retrieval accuracy for multi-document scenarios

Key Fixes:
1. Correct FAISS L2 distance score interpretation (ascending sort)
2. Smart document filtering based on proper score understanding
3. Improved text chunking with better overlap and context preservation
4. Better relevance thresholds and score normalization
5. Enhanced debug information

Installation:
sudo apt-get update
sudo apt-get install -y build-essential cmake curl python3.10-venv python3-pip libvips-dev

python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
CMAKE_ARGS="-DLLAMA_CUDA=ON" pip install llama-cpp-python --no-cache-dir --force-reinstall
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-llama-cpp llama-index-vector-stores-faiss
pip install PyMuPDF weasyprint gradio Pillow pyvips transformers accelerate faiss-cpu sentence-transformers
pip install torchao scipy
"""

import os
import re
import json
import gc
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set, Generator
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import threading
import queue
import time

warnings.filterwarnings("ignore")

# Core dependencies
import torch
import fitz # PyMuPDF
import numpy as np
from PIL import Image
import io
import gradio as gr
from tqdm import tqdm

# LlamaIndex imports
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    load_index_from_storage,
    ServiceContext
)
from llama_index.core.schema import NodeWithScore, MetadataMode, TextNode
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.base import BaseIndex

# LlamaIndex integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.faiss import FaissVectorStore

# For FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Using SimpleVectorStore instead.")

# For VLM support - Moondream 4-bit
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TextIteratorStreamer
)

# For downloading models
from huggingface_hub import hf_hub_download

# ===========================================
# LLM CONFIGURATION
# ===========================================

# Available LLM models
AVAILABLE_LLMS = {
    "gemma-3-4b": {
        "name": "Gemma 3 4B",
        "repo_id": "unsloth/gemma-3-4b-it-qat-GGUF",
        "filename": "gemma-3-4b-it-qat-Q4_K_M.gguf",
        "context_window": 20000,
        "chat_format": None,
        "supports_thinking": False
    },
    "qwen3-8b": {
        "name": "Qwen 3 8B",
        "repo_id": "unsloth/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "context_window": 20000,
        "chat_format": "chatml",
        "supports_thinking": True
    },
    "qwen3-4b": {
        "name": "Qwen 3 4B",
        "repo_id": "unsloth/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "context_window": 20000,
        "chat_format": "chatml",
        "supports_thinking": True
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "context_window": 30000,
        "chat_format": "llama-3",
        "supports_thinking": False
    },
    "gemma-3-12b": {
        "name": "Gemma 3 12B IT",
        "repo_id": "unsloth/gemma-3-12b-it-GGUF",
        "filename": "gemma-3-12b-it-Q4_K_M.gguf",
        "context_window": 10000,
        "chat_format": None,
        "supports_thinking": False
    },
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B Instruct",
        "repo_id": "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "context_window": 20000,
        "chat_format": "llama-3",
        "supports_thinking": False
    }
}

# ===========================================
# THINKING MODE HANDLER
# ===========================================

class ThinkingModeHandler:
    """Handles thinking mode for models that support it"""

    def __init__(self):
        self.thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)

    def extract_thinking(self, text: str) -> Tuple[str, str]:
        """Extract thinking content and main content"""
        thinking_matches = self.thinking_pattern.findall(text)
        thinking_content = '\n'.join(thinking_matches) if thinking_matches else ""

        # Remove thinking tags from main content
        main_content = self.thinking_pattern.sub('', text).strip()

        return thinking_content, main_content

    def append_no_think(self, prompt: str, model_config: Dict) -> str:
        """Append /no_think to prompt if model supports it"""
        if model_config.get('supports_thinking', False):
            return prompt + "/no_think"
        return prompt

# ===========================================
# DIAGNOSTICS LOGGER
# ===========================================

class DiagnosticsLogger:
    """Handles diagnostic logging for the pipeline"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.diagnostics_dir = Path("diagnostics")
        self.session_dir = None
        self.performance_metrics = {
            'pdf_processing_times': {},
            'vlm_caption_times': [],
            'index_creation_times': {},
            'summarization_times': {},
            'query_times': []
        }

        if self.enabled:
            self.diagnostics_dir.mkdir(exist_ok=True)
            self._create_session_dir()

    def _create_session_dir(self):
        """Create a directory for this session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.diagnostics_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)

    def log(self, filename: str, content: Any):
        """Log content to a file"""
        if not self.enabled or not self.session_dir:
            return

        filepath = self.session_dir / filename

        if isinstance(content, (dict, list)):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(content))

    def log_performance_metrics(self):
        """Log all performance metrics"""
        if not self.enabled:
            return

        self.log("performance_metrics.json", self.performance_metrics)

    def add_metric(self, category: str, key: str, value: float):
        """Add a performance metric"""
        if category in self.performance_metrics:
            if isinstance(self.performance_metrics[category], dict):
                self.performance_metrics[category][key] = value
            elif isinstance(self.performance_metrics[category], list):
                self.performance_metrics[category].append({key: value})

# ===========================================
# CONFIGURATION
# ===========================================

@dataclass
class EnhancedPipelineConfig:
    """Configuration for the enhanced multi-PDF pipeline"""

    # Document limits
    max_pdfs: int = 10

    # RAG chunking - IMPROVED VALUES
    rag_chunk_size: int = 2048  # Increased for better context
    rag_chunk_overlap: int = 512  # Substantial overlap for context preservation

    # Summarization - Map-Reduce settings
    summary_chunk_max_chars: int = 18000
    summary_chunk_overlap_chars: int = 3000

    map_summary_max_tokens: int = 800
    reduce_summary_max_tokens: int = 2400

    # LLM settings
    use_llama_cpp: bool = True
    current_llm_model: Optional[str] = None
    thinking_mode: bool = False
    llm_temperature: float = 0.7
    llm_chat_max_tokens: int = 8096

    # Embedding settings
    embedding_model: str = "NovaSearch/stella_en_400M_v5"
    embedding_batch_size: int = 32

    # VLM settings
    skip_vlm: bool = True
    vlm_model: str = "moondream/moondream-2b-2025-04-14-4bit"
    vlm_caption_length: str = "short"
    vlm_compile_model: bool = True
    vlm_batch_size: int = 8
    vlm_use_streaming: bool = False
    vlm_text_only_mode: bool = False

    # Query settings - IMPROVED VALUES
    similarity_top_k: int = 8  # Increased for better coverage
    comparison_top_k: int = 3

    # System
    save_diagnostics: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_streaming: bool = True
    gpu_memory_fraction: float = 0.8

# ===========================================
# PROMPT BUILDER
# ===========================================

class PromptBuilder:
    """Dynamically builds a model-specific prompt string."""
    def __init__(self, model_config: Optional[Dict] = None):
        self.model_config = model_config or {}
        self.chat_format = self.model_config.get('chat_format')

    def set_model(self, model_config: Dict):
        """Updates the builder with the current model's configuration."""
        self.model_config = model_config
        self.chat_format = self.model_config.get('chat_format')
        print(f"✅ PromptBuilder updated for chat format: '{self.chat_format or 'gemma (default)'}'")

    def get_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        """Creates the full prompt string based on the model's chat format."""
        # Llama 3 format
        if self.chat_format == 'llama-3':
            sys_prompt_str = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>" if system_prompt else "<|begin_of_text|>"
            full_prompt = f"{sys_prompt_str}<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            return full_prompt

        # ChatML (Qwen) format
        elif self.chat_format == 'chatml':
            sys_prompt_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
            full_prompt = f"{sys_prompt_str}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            return full_prompt

        # Default/Gemma format
        else:
            full_user_content = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
            return f"<start_of_turn>user\n{full_user_content}<end_of_turn>\n<start_of_turn>model\n"

# ===========================================
# STREAMING HANDLER
# ===========================================

class StreamingHandler:
    """Handles streaming output for various operations"""

    def __init__(self):
        self.queue = queue.Queue()
        self.stop_signal = False

    def put(self, text: str):
        """Add text to the streaming queue"""
        if not self.stop_signal:
            self.queue.put(text)

    def get_stream(self) -> Generator[str, None, None]:
        """Get streaming text generator"""
        accumulated = ""
        while True:
            try:
                chunk = self.queue.get(timeout=0.1)
                if chunk is None: # End of stream signal
                    break
                accumulated += chunk
                yield accumulated
            except queue.Empty:
                if self.stop_signal:
                    break
                continue

    def stop(self):
        """Stop streaming"""
        self.stop_signal = True
        self.queue.put(None) # Signal end of stream

# ===========================================
# OPTIMIZED MOONDREAM VLM HANDLER
# ===========================================

class OptimizedMoondreamHandler:
    """Optimized Moondream 4-bit handler with GPU memory management"""

    def __init__(self, config: EnhancedPipelineConfig, diagnostics_logger: DiagnosticsLogger):
        self.config = config
        self.logger = diagnostics_logger
        self.model = None
        self.compiled = False

        if not self.config.skip_vlm:
            self._load_vlm()

    def _load_vlm(self):
        """Load the 4-bit quantized Moondream model"""
        print(f"🌙 Loading Moondream 4-bit: {self.config.vlm_model}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.vlm_model,
                trust_remote_code=True,
                device_map={"": self.config.device} if self.config.device == "cuda" else "auto",
                torch_dtype=torch.float16
            )
            print("✅ Moondream 4-bit loaded successfully")
            if self.config.device == "cuda" and hasattr(self.model, 'hf_device_map'):
                print(f"Moondream model device map: {self.model.hf_device_map}")

        except Exception as e:
            print(f"❌ Failed to load Moondream: {e}")
            self.model = None

    def _compile_model_if_needed(self, num_images: int):
        """Compile model for faster inference if processing many images"""
        if (self.config.vlm_compile_model and
            not self.compiled and
            num_images >= 40 and
            self.model is not None):
            print("⚡ Compiling Moondream model for faster inference...")
            try:
                self.model.model.compile()
                self.compiled = True
                print("✅ Moondream model compiled successfully")
            except AttributeError:
                try:
                    self.model.compile()
                    self.compiled = True
                    print("✅ Moondream model compiled successfully with .compile()")
                except Exception as e_compile:
                    print(f"⚠️ Moondream compilation failed: {e_compile}. Continuing without compilation.")
            except Exception as e:
                print(f"⚠️ Moondream compilation failed: {e}. Continuing without compilation.")

    def _process_batch(self, images: List[Image.Image], batch_size: int) -> Generator[List[Image.Image], None, None]:
        """Process images in batches for memory efficiency"""
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            yield batch

    def caption_images(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for images with optimizations using documented API."""
        if not self.model or self.config.skip_vlm:
            return ["Image (VLM skipped or not loaded)" for _ in images]

        if self.config.vlm_text_only_mode:
            return self._extract_text_only(images)

        self._compile_model_if_needed(len(images))

        all_captions = []
        total_start_time = time.time()
        batch_size = self.config.vlm_batch_size
        total_batches = (len(images) + batch_size - 1) // batch_size

        with tqdm(total=len(images), desc="Captioning images with Moondream") as pbar:
            for batch_idx, image_batch_pil in enumerate(self._process_batch(images, batch_size)):
                batch_processing_start_time = time.time()
                if torch.cuda.is_available() and batch_idx > 0:
                    torch.cuda.empty_cache()

                for img_pil in image_batch_pil:
                    img_caption_start_time = time.time()
                    try:
                        if img_pil.mode != "RGB":
                            img_pil_rgb = img_pil.convert("RGB")
                        else:
                            img_pil_rgb = img_pil

                        caption_length_param = self.config.vlm_caption_length

                        current_caption_text = ""
                        if self.config.vlm_use_streaming:
                            caption_tokens = []
                            for token in self.model.caption(img_pil_rgb, length=caption_length_param, stream=True)["caption"]:
                                caption_tokens.append(token)
                            current_caption_text = "".join(caption_tokens)
                        else:
                            current_caption_text = self.model.caption(img_pil_rgb, length=caption_length_param)["caption"]

                        if caption_length_param == "short":
                            details_query = "What are the key visual elements, text, or diagrams in this image?"
                            details = self.model.query(img_pil_rgb, details_query)["answer"]
                            current_caption_text = f"{current_caption_text}. {details}".strip()

                        all_captions.append(current_caption_text if current_caption_text else "Could not generate caption.")

                        img_processing_time = time.time() - img_caption_start_time
                        self.logger.add_metric(
                            'vlm_caption_times',
                            f'image_{len(all_captions)-1}_time',
                            img_processing_time
                        )

                    except Exception as e:
                        print(f"⚠️ Error captioning an image: {e}")
                        all_captions.append("Error generating caption.")
                    pbar.update(1)

                batch_processing_time = time.time() - batch_processing_start_time
                print(f"Batch {batch_idx + 1}/{total_batches} (size {len(image_batch_pil)}) processed in {batch_processing_time:.2f}s")

        total_processing_time = time.time() - total_start_time
        avg_time_per_image = total_processing_time / len(images) if images else 0
        print(f"✅ Captioned {len(images)} images in {total_processing_time:.2f}s (avg: {avg_time_per_image:.2f}s/image)")

        self.logger.log("vlm_captioning_performance_summary.json", {
            "total_images_captioned": len(images),
            "total_captioning_time_sec": total_processing_time,
            "average_time_per_image_sec": avg_time_per_image,
            "vlm_batch_size_config": batch_size,
            "model_compiled_for_vlm": self.compiled,
            "vlm_streaming_config": self.config.vlm_use_streaming,
            "vlm_caption_length_config": self.config.vlm_caption_length
        })
        return all_captions

    def _extract_text_only(self, images: List[Image.Image]) -> List[str]:
        """Fast text-only extraction from images using documented API."""
        print("⚡ Running in VLM text-only extraction mode (Moondream query).")
        text_results = []

        for img_pil in tqdm(images, desc="Extracting text from images (Moondream query)"):
            try:
                if img_pil.mode != "RGB":
                    img_pil_rgb = img_pil.convert("RGB")
                else:
                    img_pil_rgb = img_pil

                text_query = "List visible text in max 20 words."
                text_response = self.model.query(img_pil_rgb, text_query)["answer"]

                text_results.append(f"Text from image: {text_response}" if text_response else "No text extracted by VLM query.")
            except Exception as e:
                print(f"⚠️ Error extracting text via VLM query: {e}")
                text_results.append("Error during VLM text extraction.")
        return text_results

    def unload(self):
        """Unload VLM to free memory"""
        if self.model:
            print("🗑️ Unloading Moondream VLM...")

            del self.model
            self.model = None
            self.compiled = False

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("✅ Moondream VLM unloaded and memory cleared.")
        else:
            print("ℹ️ Moondream VLM already unloaded or was not loaded.")

# ===========================================
# DOCUMENT MANAGER
# ===========================================

class DocumentManager:
    """Manages multiple PDF documents and their metadata"""
    def __init__(self, config: EnhancedPipelineConfig):
        self.config = config
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.indices: Dict[str, VectorStoreIndex] = {}
        self.raw_texts: Dict[str, List[Dict]] = {}
        self.image_data: Dict[str, List[Dict]] = {}

    def add_document(self, pdf_name: str, pdf_path: str,
                     raw_text_by_page: List[Dict],
                     images_with_captions: List[Dict],
                     index: VectorStoreIndex):
        if len(self.documents) >= self.config.max_pdfs:
            raise ValueError(f"Maximum number of PDFs ({self.config.max_pdfs}) reached")
        self.documents[pdf_name] = {
            'path': pdf_path, 'upload_time': datetime.now(),
            'page_count': len(raw_text_by_page)
        }
        self.indices[pdf_name] = index
        self.raw_texts[pdf_name] = raw_text_by_page
        self.image_data[pdf_name] = images_with_captions

    def remove_document(self, pdf_name: str):
        if pdf_name in self.documents:
            del self.documents[pdf_name]
            del self.indices[pdf_name]
            del self.raw_texts[pdf_name]
            del self.image_data[pdf_name]

    def get_all_documents(self) -> List[str]:
        return list(self.documents.keys())

    def get_document_index(self, pdf_name: str) -> Optional[VectorStoreIndex]:
        return self.indices.get(pdf_name)

    def clear_all(self):
        self.documents.clear()
        self.indices.clear()
        self.raw_texts.clear()
        self.image_data.clear()

# ===========================================
# ENHANCED PDF PROCESSOR (IMPROVED CHUNKING)
# ===========================================

class EnhancedPDFProcessor:
    """Process PDFs with LlamaIndex integration and improved chunking"""

    def __init__(self, config: EnhancedPipelineConfig, diagnostics_logger: DiagnosticsLogger):
        self.config = config
        self.logger = diagnostics_logger

    def extract_content_from_pdf(self, pdf_path: str) -> Tuple[List[Dict], List[Image.Image], List[int]]:
        """Extract text and images from PDF with optimizations"""
        print(f"📄 Processing PDF: {pdf_path}")
        start_time = time.time()
        raw_text_by_page, pil_images, image_page_numbers = [], [], []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")
            return [], [], []

        max_image_size = 1024
        min_image_size = 50

        for page_num_idx in tqdm(range(len(doc)), desc="Extracting pages"):
            page = doc[page_num_idx]
            page_num_actual = page_num_idx + 1
            text = page.get_text("text")
            if text.strip():
                raw_text_by_page.append({'page_num': page_num_actual, 'text': text})

            if not self.config.skip_vlm:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        img_width = base_image.get("width", 0)
                        img_height = base_image.get("height", 0)
                        if img_width < min_image_size or img_height < min_image_size:
                            continue

                        pil_image = Image.open(io.BytesIO(image_bytes))
                        if pil_image.mode == 'RGBA':
                            bg = Image.new('RGB', pil_image.size, (255, 255, 255))
                            bg.paste(pil_image, mask=pil_image.split()[3])
                            pil_image = bg

                        if pil_image.width > max_image_size or pil_image.height > max_image_size:
                            ratio = min(max_image_size / pil_image.width, max_image_size / pil_image.height)
                            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                        pil_images.append(pil_image)
                        image_page_numbers.append(page_num_actual)
                    except Exception as e_img:
                        print(f"⚠️ Could not extract image xref {xref} on page {page_num_actual}: {e_img}")

        doc.close()
        pdf_name = os.path.basename(pdf_path)
        extraction_time = time.time() - start_time
        self.logger.add_metric('pdf_processing_times', pdf_name, extraction_time)
        print(f"✅ Extracted {len(raw_text_by_page)} text pages and {len(pil_images)} images in {extraction_time:.2f}s")
        self.logger.log(f"extraction_details_{pdf_name}.json", {
            "total_pages": len(raw_text_by_page), "total_images": len(pil_images),
            "extraction_time": extraction_time,
            "images_per_page": {str(pn): image_page_numbers.count(pn) for pn in set(image_page_numbers)}
        })
        return raw_text_by_page, pil_images, image_page_numbers

    def chunk_for_summarization(self, raw_text_by_page: List[Dict],
                                image_captions_with_pages: List[Dict]) -> List[str]:
        """Create large chunks for map-reduce summarization"""
        print("Chunking for summarization (V1-style simple char-based)...")
        page_texts_map = {item['page_num']: item['text'] for item in raw_text_by_page}
        page_captions_map = {}
        for cap_info in image_captions_with_pages:
            p_num = cap_info['page_num']
            if p_num not in page_captions_map:
                page_captions_map[p_num] = []
            page_captions_map[p_num].append(f"(Image on page {p_num}: {cap_info['caption']})")

        full_text_for_summary = ""
        all_page_nums = sorted(list(set(page_texts_map.keys()) | set(page_captions_map.keys())))

        for p_num in all_page_nums:
            full_text_for_summary += f"[Page {p_num}]\n"
            if p_num in page_texts_map:
                full_text_for_summary += page_texts_map[p_num] + "\n"
            if p_num in page_captions_map:
                full_text_for_summary += " ".join(page_captions_map[p_num]) + "\n"
            full_text_for_summary += "\n"

        chunks = []
        if not full_text_for_summary.strip():
            return chunks

        start_idx = 0
        text_len = len(full_text_for_summary)
        max_chars = self.config.summary_chunk_max_chars
        overlap = self.config.summary_chunk_overlap_chars

        while start_idx < text_len:
            end_idx = min(start_idx + max_chars, text_len)
            current_chunk_text = full_text_for_summary[start_idx:end_idx]

            if end_idx < text_len:
                break_pos_para = current_chunk_text.rfind("\n\n")
                break_pos_sent = current_chunk_text.rfind("\n")

                if break_pos_para != -1 and break_pos_para > overlap:
                    end_idx = start_idx + break_pos_para + 2
                elif break_pos_sent != -1 and break_pos_sent > overlap:
                    end_idx = start_idx + break_pos_sent + 1

            chunks.append(full_text_for_summary[start_idx:end_idx])

            if end_idx == text_len:
                break

            start_idx += (max_chars - overlap)
            if start_idx >= end_idx:
                start_idx = end_idx

        self.logger.log("summarization_chunks.json", [
            {"chunk_index": i, "length": len(chunk), "preview": chunk[:200] + "..."}
            for i, chunk in enumerate(chunks)
        ])
        print(f"Created {len(chunks)} summarization chunks (V1-style).")
        return chunks

    def create_llamaindex_documents(self, pdf_name: str, raw_text_by_page: List[Dict],
                                    image_captions_with_pages: List[Dict]) -> List[Document]:
        """FIXED: Convert extracted content to LlamaIndex documents with improved context preservation"""
        documents = []
        
        # FIXED: Strategy 1 - Create section-based documents for better context
        current_section = ""
        current_pages = []
        section_threshold = 4000  # Characters per section
        
        for page_data in raw_text_by_page:
            page_text = page_data['text'].strip()
            page_num = page_data['page_num']
            
            # Skip very short pages (likely headers/footers only)
            if len(page_text) < 100:
                continue
            
            # Add page to current section
            if current_section:
                current_section += f"\n\n--- Page {page_num} ---\n{page_text}"
            else:
                current_section = f"--- Page {page_num} ---\n{page_text}"
            current_pages.append(page_num)
            
            # If section is getting large enough, save it
            if len(current_section) >= section_threshold:
                doc = Document(
                    text=current_section,
                    metadata={
                        'source': pdf_name,
                        'page_number': f"{current_pages[0]}-{current_pages[-1]}" if len(current_pages) > 1 else str(current_pages[0]),
                        'type': 'text',
                        'chunk_type': 'section',
                        'page_count': len(current_pages)
                    }
                )
                documents.append(doc)
                
                # Start new section with overlap (last 500 chars)
                overlap_text = current_section[-500:] if len(current_section) > 500 else current_section
                current_section = f"(continued from previous section)...\n{overlap_text}\n\n--- Page {page_num} ---\n{page_text}"
                current_pages = [page_num]
        
        # Don't forget the last section
        if current_section and len(current_section) > 200:
            doc = Document(
                text=current_section,
                metadata={
                    'source': pdf_name,
                    'page_number': f"{current_pages[0]}-{current_pages[-1]}" if len(current_pages) > 1 else str(current_pages[0]),
                    'type': 'text',
                    'chunk_type': 'section',
                    'page_count': len(current_pages)
                }
            )
            documents.append(doc)

        # FIXED: Strategy 2 - Individual page documents with enriched context
        for page_data in raw_text_by_page:
            page_text = page_data['text'].strip()
            page_num = page_data['page_num']
            
            if len(page_text) > 300:  # Only substantial pages
                # Add context from neighboring pages
                enriched_text = f"Page {page_num} Content:\n{page_text}"
                
                # Add previous page context if available
                prev_page = next((p for p in raw_text_by_page if p['page_num'] == page_num - 1), None)
                if prev_page and len(prev_page['text']) > 100:
                    prev_context = prev_page['text'][:200] + "..."
                    enriched_text = f"Previous page context: {prev_context}\n\n{enriched_text}"
                
                # Add next page context if available
                next_page = next((p for p in raw_text_by_page if p['page_num'] == page_num + 1), None)
                if next_page and len(next_page['text']) > 100:
                    next_context = next_page['text'][:200] + "..."
                    enriched_text += f"\n\nNext page context: {next_context}"
                
                doc = Document(
                    text=enriched_text,
                    metadata={
                        'source': pdf_name,
                        'page_number': page_num,
                        'type': 'text',
                        'chunk_type': 'page_with_context'
                    }
                )
                documents.append(doc)

        # FIXED: Strategy 3 - Improved image caption processing with better context
        for img_data in image_captions_with_pages:
            caption_text = img_data['caption']
            page_num = img_data['page_num']
            
            # Find the corresponding page text and surrounding context
            page_context = ""
            for i, page_data in enumerate(raw_text_by_page):
                if page_data['page_num'] == page_num:
                    # Get the full page text
                    page_context = page_data['text']
                    
                    # Also include some context from previous and next pages
                    if i > 0:
                        prev_text = raw_text_by_page[i-1]['text'][:300]
                        page_context = f"Previous page: {prev_text}...\n\n{page_context}"
                    
                    if i < len(raw_text_by_page) - 1:
                        next_text = raw_text_by_page[i+1]['text'][:300] 
                        page_context = f"{page_context}\n\nNext page: {next_text}..."
                    break
            
            # Create comprehensive image document
            enriched_text = f"Visual Content from Page {page_num}:\n{caption_text}"
            if page_context:
                enriched_text += f"\n\nPage Text Context:\n{page_context[:1000]}"  # Limit context length
            
            doc = Document(
                text=enriched_text,
                metadata={
                    'source': pdf_name,
                    'page_number': page_num,
                    'type': 'image_caption',
                    'chunk_type': 'image_with_rich_context'
                }
            )
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} LlamaIndex documents for {pdf_name} with improved context preservation")
        return documents

# ===========================================
# COMPARISON ENGINE
# ===========================================

class PDFComparisonEngine:
    """Engine for comparing similar PDFs with streaming support"""

    def __init__(self, config: EnhancedPipelineConfig, thinking_handler: ThinkingModeHandler, prompt_builder: PromptBuilder):
        self.config = config
        self.thinking_handler = thinking_handler
        self.llm = None
        self.prompt_builder = prompt_builder

    def set_llm(self, llm):
        """Set the LLM for comparison"""
        self.llm = llm

    def compare_documents_streaming(self, query: str, doc_manager: DocumentManager,
                                    selected_docs: List[str], streaming_handler: StreamingHandler) -> str:
        if len(selected_docs) < 2:
            return "Please select at least 2 documents for comparison."

        if not self.llm:
            return "LLM not loaded. Please select an LLM model first."

        streaming_handler.put("🔍 Analyzing documents...\n\n")
        doc_results = {}
        for doc_name in selected_docs:
            streaming_handler.put(f"📄 Processing {doc_name}...\n")
            index = doc_manager.get_document_index(doc_name)
            if index:
                query_engine = index.as_query_engine(similarity_top_k=self.config.comparison_top_k)
                response = query_engine.query(query)
                source_texts = []
                for node in response.source_nodes:
                    page_num = node.metadata.get('page_number', 'Unknown')
                    source_texts.append({'text': node.text, 'page': page_num, 'score': node.score})
                doc_results[doc_name] = {'response': str(response), 'sources': source_texts}

        streaming_handler.put("\n🤖 Generating comparison...\n\n")
        comparison_prompt = self._create_comparison_prompt(query, doc_results)

        if not self.config.thinking_mode and self.config.current_llm_model in AVAILABLE_LLMS:
            model_config = AVAILABLE_LLMS[self.config.current_llm_model]
            if model_config.get('supports_thinking', False):
                comparison_prompt = self.thinking_handler.append_no_think(comparison_prompt, model_config)

        final_prompt = self.prompt_builder.get_prompt(user_content=comparison_prompt)

        if self.config.enable_streaming and hasattr(self.llm, 'stream_complete'):
            full_response_text = ""
            for delta in self.llm.stream_complete(final_prompt):
                streaming_handler.put(delta.delta)
                full_response_text += delta.delta
        else:
            response = self.llm.complete(final_prompt)
            full_response_text = str(response)

            if self.config.thinking_mode and self.config.current_llm_model in AVAILABLE_LLMS:
                model_config = AVAILABLE_LLMS[self.config.current_llm_model]
                if model_config.get('supports_thinking', False):
                    thinking_content, main_content = self.thinking_handler.extract_thinking(full_response_text)
                    if thinking_content:
                        streaming_handler.put(f"💭 Thinking:\n{thinking_content}\n\n")
                    streaming_handler.put(main_content)
                else:
                    streaming_handler.put(full_response_text)
            else:
                streaming_handler.put(full_response_text)

        streaming_handler.stop()
        return "Comparison complete."

    def _create_comparison_prompt(self, query: str, doc_results: Dict) -> str:
        prompt = f"""You are an AI assistant comparing information from multiple documents.

Your task is to analyze the similarities and differences regarding the query: "{query}"

Please provide a structured comparison covering:
1. Common information or consensus found across the documents.
2. Unique points, information, or perspectives found in specific documents.
3. Any contradictions, discrepancies, or disagreements between the documents.
4. If applicable, note any evolution of guidelines, policies, or information over time if suggested by the document content.

Present your findings clearly. For each document, relevant excerpts are provided below:

"""

        for doc_name, results in doc_results.items():
            prompt += f"\n--- Document: {doc_name} ---\n"
            prompt += f"Summary regarding the query: {results['response']}\n"
            if results['sources']:
                prompt += "Key source excerpts:\n"
                for i, source in enumerate(results['sources'][:3], 1):
                    prompt += f"  {i}. (Approx. Page {source['page']}): {source['text'][:250]}...\n"
            else:
                prompt += "No specific source excerpts retrieved for this query.\n"
        prompt += "\n\nComparison Output:\n"
        return prompt

# ===========================================
# ENHANCED QUERY ENGINE (COMPLETELY FIXED)
# ===========================================

class EnhancedQueryEngine:
    """FIXED: Custom query engine with proper FAISS L2 score handling and optimized retrieval"""

    def __init__(self, config: EnhancedPipelineConfig, doc_manager: DocumentManager, thinking_handler: ThinkingModeHandler, prompt_builder: PromptBuilder):
        self.config = config
        self.doc_manager = doc_manager
        self.thinking_handler = thinking_handler
        self.prompt_builder = prompt_builder
        self.llm = None

    def set_llm(self, llm):
        """Set the LLM for queries"""
        self.llm = llm

    def query_with_sources_optimized(self, query: str, selected_docs: Optional[List[str]] = None,
                                     chat_history_str: str = "") -> Tuple[str, List[Dict], Optional[str]]:
        """FIXED: Proper FAISS L2 score handling and improved document filtering"""
        if not selected_docs:
            selected_docs = self.doc_manager.get_all_documents()

        if not selected_docs:
            return "No documents available for querying.", [], None

        if not self.llm:
            return "LLM not loaded. Please select an LLM model first.", [], None

        all_source_nodes_with_scores: List[NodeWithScore] = []
        
        # Handle thinking mode for queries
        original_query = query
        if not self.config.thinking_mode and self.config.current_llm_model in AVAILABLE_LLMS:
            model_config = AVAILABLE_LLMS[self.config.current_llm_model]
            if model_config.get('supports_thinking', False):
                query = self.thinking_handler.append_no_think(query, model_config)

        # FIXED: Collect results per document with proper score understanding
        doc_results = {}
        doc_score_stats = {}
        
        for doc_name in selected_docs:
            index = self.doc_manager.get_document_index(doc_name)
            if index:
                retriever = index.as_retriever(similarity_top_k=self.config.similarity_top_k)
                clean_query = original_query.strip()
                retrieved_nodes = retriever.retrieve(clean_query)

                # Store results per document for analysis
                doc_results[doc_name] = []
                scores = []
                
                for node_with_score in retrieved_nodes:
                    if node_with_score.score is not None:
                        # FIXED: With FAISS IndexFlatL2, lower scores = more similar
                        node_with_score.node.metadata['document_name'] = doc_name
                        doc_results[doc_name].append(node_with_score)
                        scores.append(node_with_score.score)

                # Calculate statistics for this document
                if scores:
                    doc_score_stats[doc_name] = {
                        'min_score': min(scores),  # Best match in this doc
                        'avg_score': sum(scores) / len(scores),
                        'max_score': max(scores),  # Worst match in this doc
                        'count': len(scores)
                    }
                    print(f"📊 {doc_name}: min={min(scores):.3f}, avg={sum(scores)/len(scores):.3f}, max={max(scores):.3f}")
                else:
                    doc_score_stats[doc_name] = {'min_score': float('inf'), 'avg_score': float('inf'), 'max_score': float('inf'), 'count': 0}

        # FIXED: Smart document filtering based on L2 distance (lower = better)
        if len(selected_docs) > 1 and doc_score_stats:
            # Find documents with the best (lowest) minimum scores
            valid_docs = {name: stats for name, stats in doc_score_stats.items() if stats['count'] > 0}
            
            if valid_docs:
                best_min_score = min(stats['min_score'] for stats in valid_docs.values())
                best_avg_score = min(stats['avg_score'] for stats in valid_docs.values())
                
                print(f"🎯 Best scores found: min={best_min_score:.3f}, avg={best_avg_score:.3f}")
                
                # FIXED: Filter out documents that have significantly worse scores
                # Allow documents whose best match is within reasonable range of the global best
                score_tolerance = max(0.2, best_min_score * 0.5)  # Adaptive tolerance
                
                filtered_results = {}
                for doc_name, stats in valid_docs.items():
                    doc_best_score = stats['min_score']
                    
                    # Keep documents whose best match is competitive
                    if doc_best_score <= best_min_score + score_tolerance:
                        filtered_results[doc_name] = doc_results[doc_name]
                        print(f"✅ Including {doc_name} (best score: {doc_best_score:.3f})")
                    else:
                        print(f"🚫 Filtering out {doc_name} (best score: {doc_best_score:.3f}, threshold: {best_min_score + score_tolerance:.3f})")
                
                # Only apply filtering if we keep at least one document
                if filtered_results:
                    doc_results = filtered_results
                else:
                    print("⚠️ Filtering too aggressive, keeping all documents")

        # Collect all remaining nodes
        for doc_name, nodes in doc_results.items():
            all_source_nodes_with_scores.extend(nodes)

        # Fallback if no results
        if not all_source_nodes_with_scores:
            print("⚠️ No results found, trying with relaxed parameters...")
            for doc_name in selected_docs:
                index = self.doc_manager.get_document_index(doc_name)
                if index:
                    retriever = index.as_retriever(similarity_top_k=3)  # Reduced for fallback
                    retrieved_nodes = retriever.retrieve(original_query.strip())
                    for node_with_score in retrieved_nodes:
                        node_with_score.node.metadata['document_name'] = doc_name
                        all_source_nodes_with_scores.append(node_with_score)

        if not all_source_nodes_with_scores:
            return "I couldn't find any information in the selected documents to answer your question.", [], None

        # FIXED: Sort nodes by L2 distance (ascending = best first)
        all_source_nodes_with_scores.sort(key=lambda x: x.score or float('inf'))  # Lower scores first

        # Use top N nodes overall for context
        top_k_overall = min(self.config.similarity_top_k, len(all_source_nodes_with_scores))
        context_nodes = [item.node for item in all_source_nodes_with_scores[:top_k_overall]]

        context_str_parts = []
        source_details_for_display = []

        print(f"🔍 Using {len(context_nodes)} context nodes (scores: {[f'{item.score:.3f}' for item in all_source_nodes_with_scores[:3]]})")
        
        for i, node in enumerate(context_nodes):
            doc_name = node.metadata.get('document_name', 'Unknown Document')
            page_num = node.metadata.get('page_number', 'Unknown Page')
            node_content = node.get_content(metadata_mode=MetadataMode.NONE)
            
            # Ensure we have good context length
            content_preview = node_content[:1500] if len(node_content) > 1500 else node_content
            context_str_parts.append(f"Context {i+1} from {doc_name} (Page {page_num}):\n{content_preview}")
            
            if i == 0:
                print(f"📄 Best match (score {all_source_nodes_with_scores[i].score:.3f}): {node_content[:200]}...")
            
            source_details_for_display.append({
                'document': doc_name,
                'page': page_num,
                'text': node_content[:200] + "..." if len(node_content) > 200 else node_content,
                'score': next((item.score for item in all_source_nodes_with_scores if item.node == node), float('inf'))
            })

        context_str = "\n\n---\n\n".join(context_str_parts)
        
        print(f"📝 Total context length: {len(context_str)} characters")
        print(f"📚 Sources: {list(set(node.metadata.get('document_name', '') for node in context_nodes))}")

        # ENHANCED: More direct and effective system prompt
        system_prompt = """You are a knowledgeable assistant. Use the provided context to answer the user's question comprehensively and accurately.

Instructions:
- Use the information from the context to provide a detailed, helpful answer
- If the context contains relevant information, use it to respond thoroughly
- Be specific and reference details from the context when appropriate
- If you can partially answer from the context, do so and indicate what aspects you can address"""

        user_content = f"""Context from documents:

{context_str}

Previous conversation (if any):
{chat_history_str}

Question: {original_query}

Answer based on the provided context:"""

        final_prompt_for_llm = self.prompt_builder.get_prompt(user_content, system_prompt)
        
        print(f"🤖 Querying LLM with context from {len(set(node.metadata.get('document_name', '') for node in context_nodes))} documents")

        response = self.llm.complete(final_prompt_for_llm)
        response_text = str(response)

        # Extract thinking if present
        thinking_content = None
        if self.config.thinking_mode and self.config.current_llm_model in AVAILABLE_LLMS:
            model_config = AVAILABLE_LLMS[self.config.current_llm_model]
            if model_config.get('supports_thinking', False):
                thinking_content, response_text = self.thinking_handler.extract_thinking(response_text)

        # FIXED: Better fallback handling
        if self._is_unhelpful_response(response_text) and len(context_str) > 200:
            print("⚠️ LLM gave unhelpful response despite good context. Trying alternative approach...")
            
            # Extract the most relevant content
            best_content = ""
            for node in context_nodes[:2]:  # Use top 2 most relevant
                content = node.get_content(metadata_mode=MetadataMode.NONE)
                if len(content) > 200:  # Only substantial content
                    best_content += f"From {node.metadata.get('document_name', 'document')} (Page {node.metadata.get('page_number', '?')}):\n{content}\n\n"
            
            if best_content:
                direct_prompt = f"""Based on this specific information:

{best_content}

Question: {original_query}

Provide a direct answer using the information above:"""
                
                final_direct_prompt = self.prompt_builder.get_prompt(direct_prompt)
                alternative_response = self.llm.complete(final_direct_prompt)
                alternative_text = str(alternative_response)
                
                if not self._is_unhelpful_response(alternative_text):
                    response_text = alternative_text
                    print("✅ Alternative approach successful!")

        return response_text, source_details_for_display, thinking_content

    def _is_unhelpful_response(self, response: str) -> bool:
        """Check if response is unhelpful despite having context"""
        unhelpful_phrases = [
            "couldn't find", "cannot find", "don't have information",
            "not available", "unable to locate", "no information",
            "cannot answer", "don't see", "not provided", "cannot determine"
        ]
        return any(phrase in response.lower() for phrase in unhelpful_phrases)

    def chat_streaming(self, message: str, selected_docs: List[str],
                       streaming_handler: StreamingHandler, chat_history_list: List[Tuple[str, str]]):

        # Convert chat history to string for the LLM context
        chat_history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history_list[-3:]])  # Limit history

        response_text, sources, thinking_content = self.query_with_sources_optimized(message, selected_docs, chat_history_str)

        # Stream thinking content if present
        if thinking_content:
            streaming_handler.put("💭 Thinking:\n")
            for char in thinking_content:
                streaming_handler.put(char)
                time.sleep(0.001)
            streaming_handler.put("\n\n📝 Response:\n")

        for char_or_word in response_text.split(' '): # Simple space split for streaming
            streaming_handler.put(char_or_word + ' ')
            time.sleep(0.02) # Simulate streaming delay

        if sources:
            streaming_handler.put("\n\n📚 **Sources:**\n")
            unique_sources_dict = {}
            for source in sorted(sources, key=lambda x: x['score']):  # FIXED: Sort by score ascending (best first)
                key = f"{source['document']}_p{source['page']}"
                # Store the best score for a unique source page
                if key not in unique_sources_dict or source['score'] < unique_sources_dict[key]['score']:  # FIXED: Lower is better
                    unique_sources_dict[key] = source

            for i, source in enumerate(list(unique_sources_dict.values())[:5], 1): # Top 5 unique sources
                source_entry = f"{i}. **{source['document']}** (Page {source['page']}, Score: {source['score']:.3f})\n"
                streaming_handler.put(source_entry)

        streaming_handler.stop()

# ===========================================
# MAIN APPLICATION CLASS
# ===========================================

class MultiPDFRAGApplication:
    """Main application class managing the entire pipeline"""

    def __init__(self, config: EnhancedPipelineConfig):
        self.config = config
        self.diagnostics_logger = DiagnosticsLogger(config.save_diagnostics)
        self.doc_manager = DocumentManager(config)
        self.pdf_processor = EnhancedPDFProcessor(config, self.diagnostics_logger)
        self.vlm_handler = OptimizedMoondreamHandler(config, self.diagnostics_logger) if not config.skip_vlm else None
        self.thinking_handler = ThinkingModeHandler()
        self.prompt_builder = PromptBuilder()

        self._initialize_llamaindex()

        # Initialize components that need LLM
        self.query_engine = EnhancedQueryEngine(config, self.doc_manager, self.thinking_handler, self.prompt_builder)
        self.comparison_engine = PDFComparisonEngine(config, self.thinking_handler, self.prompt_builder)

        # Model download directory
        self.models_dir = Path("./gguf_models")
        self.models_dir.mkdir(exist_ok=True)

    def _initialize_llamaindex(self):
        """Initialize LlamaIndex with improved settings"""
        print("🚀 Initializing LlamaIndex components...")

        # Use standard HuggingFaceEmbedding for the new model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model,
            device=self.config.device,
            embed_batch_size=self.config.embedding_batch_size,
            trust_remote_code=True  # Required for this model
        )

        Settings.llm = None # prevents loading the default openai api model

        # FIXED: Optimized chunking strategy for better retrieval
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]+",  # Split on sentence boundaries
            include_metadata=True,
            include_prev_next_rel=True  # Include relationships between chunks
        )

        # Global settings for better retrieval
        Settings.chunk_size = self.config.rag_chunk_size
        Settings.chunk_overlap = self.config.rag_chunk_overlap

        print(f"✅ LlamaIndex initialized with {self.config.embedding_model} and optimized chunking")

    def debug_chat_response(self, query: str, selected_docs: List[str]) -> str:
        """ENHANCED: Debug helper with better score understanding"""
        
        print(f"\n🔍 DEBUG: Processing query: '{query}'")
        print(f"📁 Selected documents: {selected_docs}")
        
        # Check if documents exist
        available_docs = self.doc_manager.get_all_documents()
        print(f"📚 Available documents: {available_docs}")
        
        if not selected_docs:
            selected_docs = available_docs
            print(f"🔄 Using all available documents: {selected_docs}")
        
        # Test retrieval for each document
        for doc_name in selected_docs:
            index = self.doc_manager.get_document_index(doc_name)
            if index:
                print(f"\n📖 Testing retrieval from {doc_name}:")
                retriever = index.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(query)
                
                # FIXED: Sort nodes properly (ascending for L2 distance)
                nodes.sort(key=lambda x: x.score or float('inf'))
                
                for i, node_with_score in enumerate(nodes):
                    print(f"  Node {i+1}: Score={node_with_score.score:.3f} (lower=better for L2 distance)")
                    content_preview = node_with_score.node.get_content()[:300] + "..."
                    print(f"  Content: {content_preview}")
                    print(f"  Metadata: {node_with_score.node.metadata}")
            else:
                print(f"❌ No index found for {doc_name}")
        
        return "Debug complete - check console output. Note: Lower scores = better matches with FAISS L2 distance."

    # [Rest of the methods remain the same but with improved error handling]
    def load_llm(self, model_key: str, thinking_mode: bool = False, progress=gr.Progress()) -> str:
        """Load a specific LLM model"""
        if model_key not in AVAILABLE_LLMS:
            return f"Unknown model: {model_key}"

        try:
            # Unload current LLM if exists
            if Settings.llm:
                print("🗑️ Unloading current LLM...")
                llm_to_unload = Settings.llm
                Settings.llm = None
                del llm_to_unload

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_config = AVAILABLE_LLMS[model_key]
            progress(0.1, desc=f"Loading {model_config['name']}...")

            # Update config
            self.config.current_llm_model = model_key
            self.config.thinking_mode = thinking_mode

            # Download model if needed
            model_path = self._download_gguf_model(model_config['repo_id'], model_config['filename'], progress)

            progress(0.7, desc="Initializing model...")

            # Get context window for current model
            context_window = model_config['context_window']

            # Initialize LLM with appropriate settings
            llm_kwargs = {
                "model_path": model_path,
                "temperature": self.config.llm_temperature,
                "max_new_tokens": self.config.llm_chat_max_tokens,
                "context_window": context_window,
                "model_kwargs": {
                    "n_gpu_layers": -1, # Use all GPU layers
                    "verbose": True
                },
                "verbose": True
            }

            # Add chat format if specified
            if model_config.get('chat_format'):
                llm_kwargs['model_kwargs']['chat_format'] = model_config['chat_format']

            Settings.llm = LlamaCPP(**llm_kwargs)

            # Update components with new LLM
            self.prompt_builder.set_model(model_config)
            self.query_engine.set_llm(Settings.llm)
            self.comparison_engine.set_llm(Settings.llm)

            progress(1.0, desc="Complete!")

            thinking_info = ""
            if model_config.get('supports_thinking', False):
                thinking_info = f" (Thinking mode: {'Enabled' if thinking_mode else 'Disabled'})"

            return f"✅ Successfully loaded {model_config['name']}{thinking_info}"

        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error loading model: {str(e)}"

    def _download_gguf_model(self, repo_id: str, filename: str, progress=gr.Progress()) -> str:
        """Download GGUF model if needed"""
        model_path = self.models_dir / filename

        if not model_path.exists():
            print(f"⬇️ Downloading GGUF model: {filename}")
            progress(0.3, desc="Downloading model...")

            model_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            progress(0.6, desc="Model downloaded...")
            return model_path_str
        else:
            progress(0.6, desc="Model found locally...")

        return str(model_path)

    def _ensure_vlm_loaded(self):
        if not self.config.skip_vlm and self.vlm_handler is None:
            print("🔄 Reinitializing VLM...")
            self.vlm_handler = OptimizedMoondreamHandler(self.config, self.diagnostics_logger)
        elif not self.config.skip_vlm and self.vlm_handler and self.vlm_handler.model is None:
            print("🔄 VLM handler exists but model not loaded. Reinitializing VLM...")
            self.vlm_handler._load_vlm()

    def process_pdfs_batch(self, pdf_file_paths: List[str], progress=gr.Progress()) -> Tuple[str, List[str]]:
        if not pdf_file_paths:
            return "No files uploaded", []

        # Check if LLM is loaded
        if not Settings.llm:
            return "⚠️ Please select and load an LLM model before processing PDFs", []

        results = []
        total_files = len(pdf_file_paths)
        vlm_reinit_interval = 3

        for i, pdf_path_obj in enumerate(pdf_file_paths):
            pdf_path_str = pdf_path_obj if isinstance(pdf_path_obj, str) else pdf_path_obj.name
            pdf_name = os.path.basename(pdf_path_str)

            progress((i + 1) / total_files, desc=f"Processing {pdf_name}...")

            if (i > 0 and i % vlm_reinit_interval == 0 and not self.config.skip_vlm and self.vlm_handler):
                print("♻️ Reinitializing VLM for memory optimization during batch processing...")
                self.vlm_handler.unload()
                self.vlm_handler = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                self._ensure_vlm_loaded()

            try:
                class TempFileWrapper:
                    def __init__(self, path_str):
                        self.name = path_str

                status, _ = self.process_pdf(TempFileWrapper(pdf_path_str), progress_subtask=True)
                results.append(status)
            except Exception as e:
                results.append(f"❌ Error processing {pdf_name}: {str(e)}")
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        if self.vlm_handler and not self.config.skip_vlm:
            print("🧹 Unloading VLM after batch processing to free memory...")
            self.vlm_handler.unload()
            self.vlm_handler = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        self.diagnostics_logger.log_performance_metrics()
        summary_msg = f"Processed {total_files} files:\n" + "\n".join(results)
        return summary_msg, self.doc_manager.get_all_documents()

    def process_pdf(self, pdf_file_obj, progress=gr.Progress(), progress_subtask=False) -> Tuple[str, List[str]]:
        if not pdf_file_obj:
            return "No file uploaded", []

        # Check if LLM is loaded
        if not Settings.llm:
            return "⚠️ Please select and load an LLM model before processing PDFs", []

        pdf_path = pdf_file_obj.name
        pdf_name = os.path.basename(pdf_path)

        if pdf_name in self.doc_manager.documents:
            return f"'{pdf_name}' is already processed", self.doc_manager.get_all_documents()
        if len(self.doc_manager.documents) >= self.config.max_pdfs:
            return f"Max {self.config.max_pdfs} PDFs. Remove some.", self.doc_manager.get_all_documents()

        if not progress_subtask: progress(0.1, desc=f"Processing {pdf_name}...")

        raw_text_by_page, images, image_page_nums = self.pdf_processor.extract_content_from_pdf(pdf_path)
        if not raw_text_by_page and not images:
            return f"No content from '{pdf_name}'", self.doc_manager.get_all_documents()

        if not progress_subtask: progress(0.3, desc="Image captions..." if images else "Processing text...")

        image_captions_with_pages_for_rag_and_summary = []
        if images and not self.config.skip_vlm:
            try:
                self._ensure_vlm_loaded()
            except Exception as e:
                print(f"⚠️ VLM ensure_loaded_failed: {e}")
                self.vlm_handler = None

            if self.vlm_handler and self.vlm_handler.model:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                try:
                    captions = self.vlm_handler.caption_images(images)
                    for i, caption_text in enumerate(captions):
                        image_captions_with_pages_for_rag_and_summary.append({
                            'page_num': image_page_nums[i], 'caption': caption_text
                        })
                    self.diagnostics_logger.log(f"image_captions_{pdf_name}.json", image_captions_with_pages_for_rag_and_summary)
                except Exception as e_cap:
                    print(f"⚠️ Error during image captioning for {pdf_name}: {e_cap}")
                    for i in range(len(images)):
                        image_captions_with_pages_for_rag_and_summary.append({
                            'page_num': image_page_nums[i], 'caption': "Image processing failed"
                        })
            else:
                print(f"⚠️ VLM not available or model not loaded, skipping captions for {pdf_name}")
                for i in range(len(images)):
                    image_captions_with_pages_for_rag_and_summary.append({
                        'page_num': image_page_nums[i], 'caption': "Image (VLM not available/failed)"
                    })

        if not progress_subtask: progress(0.6, desc="Creating document index...")

        documents_for_index = self.pdf_processor.create_llamaindex_documents(
            pdf_name, raw_text_by_page, image_captions_with_pages_for_rag_and_summary
        )

        start_time_idx = time.time()
        if FAISS_AVAILABLE:
            # Get embedding dimension dynamically
            d = Settings.embed_model._model.get_sentence_embedding_dimension()
            faiss_index = faiss.IndexFlatL2(d)  # L2 distance (lower = more similar)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents_for_index, storage_context=storage_context, show_progress=not progress_subtask
            )
            print(f"✅ Created FAISS index with L2 distance (dimension: {d})")
        else:
            index = VectorStoreIndex.from_documents(documents_for_index, show_progress=not progress_subtask)

        self.diagnostics_logger.add_metric('index_creation_times', pdf_name, time.time() - start_time_idx)
        if not progress_subtask: progress(0.9, desc="Finalizing...")

        self.doc_manager.add_document(
            pdf_name, pdf_path, raw_text_by_page,
            image_captions_with_pages_for_rag_and_summary,
            index
        )
        if not progress_subtask: progress(1.0, desc="Complete!")

        llm_info = f" (LLM: {AVAILABLE_LLMS[self.config.current_llm_model]['name']})" if self.config.current_llm_model else ""
        return f"✅ '{pdf_name}' processed{llm_info}.", self.doc_manager.get_all_documents()

    def summarize_document_mapreduce(self, selected_pdf: str, progress=gr.Progress(),
                                     streaming_handler: Optional[StreamingHandler] = None) -> str:
        if not selected_pdf or selected_pdf not in self.doc_manager.documents:
            return "Please select a valid PDF to summarize"

        if not Settings.llm:
            return "⚠️ Please select and load an LLM model before summarizing"

        try:
            if self.vlm_handler and self.vlm_handler.model:
                print("🧹 Unloading VLM before summarization to free memory...")
                self.vlm_handler.unload()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("🧠 GPU memory cleared. Starting summarization...")

            progress(0.1, desc=f"Preparing to summarize {selected_pdf}...")
            start_time = time.time()

            raw_texts = self.doc_manager.raw_texts.get(selected_pdf, [])
            image_data_for_summary = self.doc_manager.image_data.get(selected_pdf, [])

            if not raw_texts: return "No text content available for summarization"

            chunks = self.pdf_processor.chunk_for_summarization(raw_texts, image_data_for_summary)
            if not chunks: return "Could not prepare chunks for summarization"

            if streaming_handler: streaming_handler.put(f"📊 Created {len(chunks)} chunks for map-reduce summarization\n\n")
            progress(0.2, desc=f"Created {len(chunks)} chunks...")

            chunk_summaries_content = []

            current_model_config = AVAILABLE_LLMS.get(self.config.current_llm_model, {})
            context_window = current_model_config.get('context_window', 8192)

            original_max_tokens = Settings.llm.max_new_tokens
            Settings.llm.max_new_tokens = self.config.map_summary_max_tokens

            # --- MAP PHASE ---
            for i, chunk_text_full in enumerate(chunks):
                map_progress = 0.2 + (0.6 * (i + 1) / len(chunks))
                progress(map_progress, desc=f"Summarizing chunk {i+1}/{len(chunks)}...")
                if streaming_handler: streaming_handler.put(f"📝 Summarizing chunk {i+1}/{len(chunks)}...\n")

                PROMPT_BUFFER_CHARS = 2048
                available_space_for_chunk = context_window - PROMPT_BUFFER_CHARS - Settings.llm.max_new_tokens
                chunk_text_to_summarize = chunk_text_full[:max(0, available_space_for_chunk)]

                map_system_prompt = f"""You are a helpful AI assistant specialized in summarizing document segments.

Please generate a comprehensive and detailed summary of the following text segment. This summary should capture ALL the key points, main arguments and any significant conclusions presented *within this segment only*.

Maintain a neutral, objective, and informative tone.

The target length for this segment's summary is approximately {self.config.map_summary_max_tokens // 4} words.

Do not add any introductory or concluding phrases that are not part of the summary content itself (e.g., avoid 'Here is the summary:').

Begin the summary directly."""

                map_user_content = f"""TEXT SEGMENT TO SUMMARIZE:

-------------------------

{chunk_text_to_summarize}

-------------------------

End of TEXT SEGMENT. Begin the summary directly."""

                map_prompt = self.prompt_builder.get_prompt(map_user_content, map_system_prompt)

                if not self.config.thinking_mode and current_model_config.get('supports_thinking', False):
                    map_prompt = self.thinking_handler.append_no_think(map_prompt, current_model_config)

                try:
                    summary_obj = Settings.llm.complete(map_prompt)
                    summary_text = str(summary_obj).strip()

                    if self.config.thinking_mode and current_model_config.get('supports_thinking', False):
                        _, summary_text = self.thinking_handler.extract_thinking(summary_text)

                    chunk_summaries_content.append(summary_text)
                    if i % 3 == 0 and torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception as e_map:
                    print(f"❌ Error summarizing chunk {i} for {selected_pdf}: {e_map}")
                    chunk_summaries_content.append(f"Error summarizing chunk {i+1}.")

            Settings.llm.max_new_tokens = original_max_tokens

            self.diagnostics_logger.log(f"map_chunk_summaries_text_{selected_pdf}.json", chunk_summaries_content)
            if not chunk_summaries_content: return "Failed to generate summaries for any chunk."

            progress(0.9, desc="Combining chunk summaries...")
            if streaming_handler: streaming_handler.put("\n🔄 Combining chunk summaries into final document summary...\n\n")

            combined_chunk_summaries_text = "\n\n---\nSegment Summary Boundary\n---\n\n".join(chunk_summaries_content)
            self.diagnostics_logger.log(f"combined_chunk_summaries_text_{selected_pdf}.txt", combined_chunk_summaries_text)

            final_summary_text = ""

            # --- REDUCE PHASE ---
            if len(combined_chunk_summaries_text) > 35000:
                print(f"Combined summaries for {selected_pdf} is long. Generating sectioned summary.")
                final_summary_text = f"# Summary of {selected_pdf}\n\n"
                final_summary_text += "The document is extensive. Here are the key points from each section:\n\n"
                if streaming_handler: streaming_handler.put(final_summary_text)

                for i, summary_item_text in enumerate(chunk_summaries_content):
                    section = f"## Section {i+1}\n\n{summary_item_text}\n\n"
                    final_summary_text += section
                    if streaming_handler:
                        streaming_handler.put(section)
                        time.sleep(0.01)
            else:
                print(f"Combined summaries for {selected_pdf} is manageable. Generating synthesized summary.")

                Settings.llm.max_new_tokens = self.config.reduce_summary_max_tokens

                reduce_system_prompt = f"""You are an expert AI assistant tasked with synthesizing multiple summaries of document segments into a single, comprehensive, and coherent final summary.

Your primary objective is to integrate the information from the provided segment summaries, eliminate redundancy, and produce a well-structured final document that accurately reflects the core content of the original source, based *only* on the summaries provided.

**Output Requirements for the Final Consolidated Summary:**

1.  **Content Focus:** Integrate ALL the information, key themes, arguments, and conclusions from the provided segment summaries.

2.  **Structure & Formatting:** The output MUST be in well-structured Markdown format and include the following distinct sections:

    a.  **Main Title:** A suitable H1 or H2 Markdown heading for the overall summary.

    b.  **Introductory Paragraph:** A brief introduction that outlines the main topic and overall scope of the original document.

    c.  **Key Terms and Descriptions Section:** A dedicated section with a subheading (e.g., `## Key Terms and Concepts`) that lists and describes significant key terms.

    d.  **Main Body:** The core of the summary, presenting a logical and flowing narrative.

    e.  **Concluding Paragraph:** A final paragraph that wraps up the main points.

3.  **No Extraneous Text:** The output MUST ONLY be the summary itself. Do NOT include any conversational text.

4.  **Target Length:** Aim for a total output length of approximately {(self.config.reduce_summary_max_tokens // 4) - 50} words.

Begin the final consolidated summary directly with the Main Title."""

                PROMPT_BUFFER_CHARS = 3072
                available_space_for_combined = context_window - PROMPT_BUFFER_CHARS - Settings.llm.max_new_tokens
                combined_summaries_for_reduce = combined_chunk_summaries_text[:max(0, available_space_for_combined)]

                reduce_user_content = f"""**Input: Collection of Segment Summaries:**

---------------------------------------

{combined_summaries_for_reduce}

---------------------------------------"""

                reduce_prompt = self.prompt_builder.get_prompt(reduce_user_content, reduce_system_prompt)

                if not self.config.thinking_mode and current_model_config.get('supports_thinking', False):
                    reduce_prompt = self.thinking_handler.append_no_think(reduce_prompt, current_model_config)

                try:
                    if streaming_handler and hasattr(Settings.llm, 'stream_complete'):
                        temp_final_summary = ""
                        for delta in Settings.llm.stream_complete(reduce_prompt):
                            streaming_handler.put(delta.delta)
                            temp_final_summary += delta.delta
                        final_summary_text = temp_final_summary

                        if self.config.thinking_mode and current_model_config.get('supports_thinking', False):
                            thinking_content, main_content = self.thinking_handler.extract_thinking(final_summary_text)
                            if thinking_content:
                                streaming_handler.queue.queue.clear()
                                streaming_handler.put(f"💭 Thinking:\n{thinking_content}\n\n")
                                streaming_handler.put(main_content)
                                final_summary_text = main_content
                    else:
                        summary_obj = Settings.llm.complete(reduce_prompt)
                        response_text = str(summary_obj).strip()

                        if self.config.thinking_mode and current_model_config.get('supports_thinking', False):
                            _, final_summary_text = self.thinking_handler.extract_thinking(response_text)
                        else:
                            final_summary_text = response_text

                        if streaming_handler: streaming_handler.put(final_summary_text)
                except Exception as e_reduce:
                    print(f"❌ Error in reduce phase for {selected_pdf}: {e_reduce}")
                    final_summary_text = f"Error generating final summary: {str(e_reduce)}\n\n---\nCombined Chunk Summaries:\n{combined_chunk_summaries_text}"
                    if streaming_handler: streaming_handler.put(final_summary_text)

            Settings.llm.max_new_tokens = original_max_tokens

            self.diagnostics_logger.log(f"final_summary_{selected_pdf}.md", final_summary_text)
            self.diagnostics_logger.add_metric('summarization_times', selected_pdf, time.time() - start_time)
            progress(1.0, desc="Summary complete!")
            if streaming_handler: streaming_handler.stop()

            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return final_summary_text

        except Exception as e_sum:
            error_msg = f"❌ Summarization failed for {selected_pdf}: {str(e_sum)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if streaming_handler:
                streaming_handler.put(error_msg)
                streaming_handler.stop()
            return error_msg

    def chat_with_documents(self, message: str, chat_history: List[Tuple[str, str]],
                            selected_docs: List[str]) -> Tuple[str, List[Tuple[str, str]], Optional[str]]:
        if not message.strip():
            return "", chat_history, None
        if not self.doc_manager.get_all_documents():
            chat_history.append((message, "⚠️ Please upload and process PDF files first."))
            return "", chat_history, None

        # Check if LLM is loaded
        if not Settings.llm:
            chat_history.append((message, "⚠️ Please select and load an LLM model first."))
            return "", chat_history, None

        docs_to_query = selected_docs if selected_docs else self.doc_manager.get_all_documents()
        query_start = time.time()

        chat_history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])

        response, sources, thinking_content = self.query_engine.query_with_sources_optimized(message, docs_to_query, chat_history_str)

        self.diagnostics_logger.add_metric('query_times', f"query_{len(self.diagnostics_logger.performance_metrics['query_times'])}",
                                             time.time() - query_start)
        if sources:
            source_text = "\n\n📚 **Sources:**\n"
            unique_sources = {}
            for source in sorted(sources, key=lambda x: x['score']):  # FIXED: Sort by score ascending (best first)
                key = f"{source['document']}_p{source['page']}"
                if key not in unique_sources or source['score'] < unique_sources[key]['score']:  # FIXED: Lower is better
                    unique_sources[key] = source

            for i, source in enumerate(list(unique_sources.values())[:5],1):
                source_text += f"{i}. **{source['document']}** (Page {source['page']}, Score: {source['score']:.3f})\n"
            response += source_text

        chat_history.append((message, response))
        return "", chat_history, thinking_content

    def compare_documents(self, query: str, selected_docs: List[str], progress=gr.Progress()) -> str:
        if not query.strip(): return "Please enter a question for comparison"
        if len(selected_docs) < 2: return "Please select at least 2 documents for comparison"

        if not Settings.llm:
            return "⚠️ Please select and load an LLM model first"

        if self.vlm_handler and self.vlm_handler.model:
            print("🧹 Unloading VLM before comparison...")
            self.vlm_handler.unload()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        progress(0.3, desc="Analyzing documents...")

        temp_stream_handler = StreamingHandler()

        thread = threading.Thread(
            target=self.comparison_engine.compare_documents_streaming,
            args=(query, self.doc_manager, selected_docs, temp_stream_handler)
        )
        thread.start()

        full_comparison_result = ""
        for chunk in temp_stream_handler.get_stream():
            full_comparison_result = chunk

        thread.join()

        progress(1.0, desc="Comparison complete!")
        return full_comparison_result

# ===========================================
# GRADIO INTERFACE
# ===========================================

def create_gradio_interface():
    """Create the Gradio interface with enhanced features"""
    config = EnhancedPipelineConfig()
    app = MultiPDFRAGApplication(config)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 FIXED Multi-PDF RAG System v9.3")
        gr.Markdown("**FIXED:** Proper FAISS L2 distance handling, improved document filtering, and better context preservation")
        uploaded_docs_state = gr.State([])
        thinking_display = gr.State("")

        with gr.Row():
            with gr.Column(scale=1): # Left Panel
                # LLM Selection Section
                gr.Markdown("### 🤖 LLM Selection")

                with gr.Row():
                    llm_dropdown = gr.Dropdown(
                        label="Select LLM Model",
                        choices=[(config['name'], key) for key, config in AVAILABLE_LLMS.items()],
                        value=None,
                        interactive=True
                    )

                thinking_mode_checkbox = gr.Checkbox(
                    label="Enable Thinking Mode (Qwen models only)",
                    value=False,
                    visible=False
                )

                load_llm_btn = gr.Button("🔄 Load LLM", variant="primary")
                llm_status = gr.Textbox(
                    label="LLM Status",
                    value="No LLM loaded",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("### 📁 Document Management")
                pdf_upload = gr.File(
                    label="Upload PDFs (Select Multiple)", file_types=[".pdf"],
                    type="filepath", file_count="multiple"
                )
                with gr.Row():
                    upload_btn = gr.Button("📤 Process PDFs", variant="primary")
                upload_status = gr.Textbox(label="Status", value="Ready...", interactive=False, lines=3)

                if torch.cuda.is_available():
                    gpu_status = gr.Textbox(label="GPU Memory", value="N/A", interactive=False, lines=2)
                    clear_gpu_btn = gr.Button("🧹 Clear GPU Memory")

                doc_list_for_selection = gr.CheckboxGroup(
                    label=f"Select Processed Documents (Max {config.max_pdfs}) for Removal/Actions",
                    choices=[], value=[], interactive=True
                )
                with gr.Row():
                    remove_selected_btn = gr.Button("Remove Selected Docs", variant="stop")

                gr.Markdown("### 🔧 Settings")
                enable_diagnostics_cb = gr.Checkbox(label="Enable Diagnostics", value=config.save_diagnostics)
                vlm_mode_radio = gr.Radio(
                    label="Image Processing Mode",
                    choices=[
                        ("Skip (fastest)", "skip"),
                        ("Text only (fast)", "text_only"),
                        ("Full captioning (comprehensive)", "full")
                    ],
                    value="skip" if not config.skip_vlm else "skip"
                )

            with gr.Column(scale=2): # Right Panel (Tabs)
                with gr.Tabs():
                    with gr.TabItem("💬 Chat with PDFs"):
                        chat_doc_selector = gr.CheckboxGroup(
                            label="Chat with (select from processed documents, empty for all):",
                            choices=[], value=[], interactive=True
                        )

                        # FIXED: Enhanced Debug Section
                        with gr.Accordion("🔧 Debug Mode (FIXED L2 Distance)", open=False):
                            gr.Markdown("**FIXED:** Now properly handles FAISS L2 distance scores (lower = better)")
                            debug_query_box = gr.Textbox(
                                label="Debug Query (or type 'debug: your question' in chat):", 
                                placeholder="Enter query to debug retrieval...", 
                                lines=1
                            )
                            debug_btn = gr.Button("🔍 Debug Retrieval", variant="secondary")
                            debug_output = gr.Textbox(
                                label="Debug Results", 
                                value="", 
                                interactive=False, 
                                lines=10,
                                visible=False
                            )

                        # Thinking display for chat
                        chat_thinking_display = gr.Textbox(
                            label="💭 Thinking Process",
                            value="",
                            interactive=False,
                            lines=5,
                            visible=False
                        )

                        chatbot_display = gr.Chatbot(label="Chat History", height=450, bubble_full_width=False)
                        chat_msg_box = gr.Textbox(label="Your Question:", placeholder="Ask about the documents...", lines=2)
                        with gr.Row():
                            chat_submit_btn = gr.Button("Send", variant="primary")
                            chat_clear_btn = gr.Button("Clear Chat")
                            stream_chat_cb = gr.Checkbox(label="Stream Response", value=config.enable_streaming)

                    with gr.TabItem("📝 Summarize Document"):
                        summary_doc_selector = gr.Dropdown(
                            label="Select a document to summarize:", choices=[], value=None, interactive=True
                        )
                        with gr.Row():
                            summarize_btn = gr.Button("Generate Summary", variant="primary")
                            stream_summary_cb = gr.Checkbox(label="Stream Summary", value=config.enable_streaming)
                        summary_output_md = gr.Markdown(label="Document Summary", value="Summary will appear here...")

                    with gr.TabItem("🔍 Compare Documents"):
                        compare_doc_selector = gr.CheckboxGroup(
                            label="Select documents to compare (minimum 2):", choices=[], value=[], interactive=True
                        )
                        compare_query_box = gr.Textbox(label="Comparison Question:", placeholder="e.g., What are the differences in methodology?", lines=2)
                        with gr.Row():
                            compare_btn = gr.Button("Compare Documents", variant="primary")
                            stream_compare_cb = gr.Checkbox(label="Stream Comparison", value=config.enable_streaming)
                        comparison_output_md = gr.Markdown(label="Comparison Results", value="Comparison results will appear here...")

        # Event Handlers & Logic

        def get_gpu_status_str():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"Alloc: {allocated:.2f}GB | Reserv: {reserved:.2f}GB | Total: {total:.2f}GB"
            return "GPU not available"

        def update_all_doc_selectors(processed_doc_names: List[str]):
            chat_selection = processed_doc_names if processed_doc_names else []
            summary_selection = processed_doc_names[0] if processed_doc_names else None

            return (
                gr.CheckboxGroup(choices=processed_doc_names, value=[]),
                gr.CheckboxGroup(choices=processed_doc_names, value=chat_selection),
                gr.Dropdown(choices=processed_doc_names, value=summary_selection),
                gr.CheckboxGroup(choices=processed_doc_names, value=[])
            )

        # LLM Event Handlers
        def handle_llm_selection(llm_key):
            """Handle LLM selection and show/hide thinking mode"""
            if llm_key and llm_key in AVAILABLE_LLMS:
                model_config = AVAILABLE_LLMS[llm_key]
                if model_config.get('supports_thinking', False):
                    return gr.Checkbox(visible=True, value=False)
                else:
                    return gr.Checkbox(visible=False, value=False)
            return gr.Checkbox(visible=False, value=False)

        def handle_load_llm(llm_key, thinking_mode):
            """Handle loading of selected LLM"""
            if not llm_key:
                return "Please select an LLM model first", gr.Textbox(visible=False)

            status = app.load_llm(llm_key, thinking_mode)

            show_thinking = False
            if llm_key in AVAILABLE_LLMS and AVAILABLE_LLMS[llm_key].get('supports_thinking', False) and thinking_mode:
                show_thinking = True

            return status, gr.Textbox(visible=show_thinking)

        def handle_pdf_upload_btn_click(list_of_pdf_filepaths, current_doc_state, progress=gr.Progress()):
            if not list_of_pdf_filepaths:
                return "No files provided for processing.", current_doc_state, *update_all_doc_selectors(current_doc_state), \
                       (get_gpu_status_str() if torch.cuda.is_available() else None)

            new_files_to_process = []
            for f_path in list_of_pdf_filepaths:
                if os.path.basename(f_path) not in current_doc_state:
                    new_files_to_process.append(f_path)

            if not new_files_to_process:
                return "All selected files already seem to be processed or no new files found.", current_doc_state, *update_all_doc_selectors(current_doc_state), \
                       (get_gpu_status_str() if torch.cuda.is_available() else None)

            status_msg, updated_doc_names_state = app.process_pdfs_batch(new_files_to_process, progress=progress)

            gpu_s = get_gpu_status_str() if torch.cuda.is_available() else None
            outputs_for_selectors = update_all_doc_selectors(updated_doc_names_state)

            if gpu_s:
                return status_msg, updated_doc_names_state, *outputs_for_selectors, gpu_s
            return status_msg, updated_doc_names_state, *outputs_for_selectors

        # LLM handlers
        llm_dropdown.change(
            fn=handle_llm_selection,
            inputs=[llm_dropdown],
            outputs=[thinking_mode_checkbox]
        )

        load_llm_btn.click(
            fn=handle_load_llm,
            inputs=[llm_dropdown, thinking_mode_checkbox],
            outputs=[llm_status, chat_thinking_display]
        )

        upload_btn.click(
            fn=handle_pdf_upload_btn_click,
            inputs=[pdf_upload, uploaded_docs_state],
            outputs=(
                [upload_status, uploaded_docs_state, doc_list_for_selection,
                 chat_doc_selector, summary_doc_selector, compare_doc_selector] +
                ([gpu_status] if torch.cuda.is_available() else [])
            )
        )

        def handle_remove_selected_docs_click(docs_to_remove_names, current_doc_state):
            if not docs_to_remove_names:
                return "No documents selected for removal.", current_doc_state, *update_all_doc_selectors(current_doc_state), \
                       (get_gpu_status_str() if torch.cuda.is_available() else None)

            removed_count = 0
            for doc_name in docs_to_remove_names:
                if doc_name in current_doc_state:
                    app.doc_manager.remove_document(doc_name)
                    removed_count +=1

            updated_doc_list = app.doc_manager.get_all_documents()
            status = f"Removed {removed_count} document(s)."
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gpu_s = get_gpu_status_str() if torch.cuda.is_available() else None
            outputs_for_selectors = update_all_doc_selectors(updated_doc_list)
            if gpu_s:
                return status, updated_doc_list, *outputs_for_selectors, gpu_s
            return status, updated_doc_list, *outputs_for_selectors

        remove_selected_btn.click(
            fn=handle_remove_selected_docs_click,
            inputs=[doc_list_for_selection, uploaded_docs_state],
            outputs=(
                [upload_status, uploaded_docs_state, doc_list_for_selection,
                 chat_doc_selector, summary_doc_selector, compare_doc_selector] +
                ([gpu_status] if torch.cuda.is_available() else [])
            )
        )

        # FIXED: Debug handler
        def handle_debug_click(debug_query, selected_docs):
            if not debug_query.strip():
                return "Please enter a query to debug", gr.Textbox(visible=False)
            
            debug_result = app.debug_chat_response(debug_query.strip(), selected_docs)
            return debug_result, gr.Textbox(visible=True)

        debug_btn.click(
            fn=handle_debug_click,
            inputs=[debug_query_box, chat_doc_selector],
            outputs=[debug_output, debug_output]
        )

        # FIXED: Chat handler with debug support
        def handle_chat_submit(message: str, history: List[Tuple[str,str]],
                               selected_chat_docs: List[str], stream_chat: bool):
            if not message.strip():
                yield message, history, ""
                return

            # ADDED: Debug check
            if message.lower().startswith("debug:"):
                debug_query = message[6:].strip()  # Remove "debug:" prefix
                debug_result = app.debug_chat_response(debug_query, selected_chat_docs)
                history.append((message, debug_result))
                yield "", history, ""
                return

            if not app.doc_manager.get_all_documents():
                history.append((message, "⚠️ Please upload and process PDF files first."))
                yield "", history, ""
                return

            if stream_chat and config.enable_streaming:
                streaming_handler = StreamingHandler()
                thread = threading.Thread(target=app.query_engine.chat_streaming,
                                          args=(message, selected_chat_docs, streaming_handler, history))
                thread.start()

                current_response = ""
                history.append((message, current_response))
                for chunk in streaming_handler.get_stream():
                    current_response = chunk
                    history[-1] = (message, current_response)
                    yield "", history, ""
                thread.join()
            else:
                _, updated_history, thinking_content = app.chat_with_documents(message, history, selected_chat_docs)
                yield "", updated_history, thinking_content or ""

        chat_submit_btn.click(handle_chat_submit,
                              [chat_msg_box, chatbot_display, chat_doc_selector, stream_chat_cb],
                              [chat_msg_box, chatbot_display, chat_thinking_display])
        chat_msg_box.submit(handle_chat_submit,
                            [chat_msg_box, chatbot_display, chat_doc_selector, stream_chat_cb],
                            [chat_msg_box, chatbot_display, chat_thinking_display])
        chat_clear_btn.click(lambda: ([], "", ""), outputs=[chatbot_display, chat_msg_box, chat_thinking_display])

        # Summarization handler
        def handle_summarize_btn_click(doc_to_summarize_name: str, stream_summary: bool, progress=gr.Progress()):
            if not doc_to_summarize_name:
                yield "Please select a document to summarize."
                return

            if stream_summary and config.enable_streaming:
                streaming_handler = StreamingHandler()
                thread = threading.Thread(target=app.summarize_document_mapreduce,
                                          args=(doc_to_summarize_name, progress, streaming_handler))
                thread.start()
                current_summary = ""
                for chunk in streaming_handler.get_stream():
                    current_summary = chunk
                    yield current_summary
                thread.join()
            else:
                summary_result = app.summarize_document_mapreduce(doc_to_summarize_name, progress=progress)
                yield summary_result

        summarize_btn.click(handle_summarize_btn_click,
                            [summary_doc_selector, stream_summary_cb],
                            [summary_output_md])

        # Comparison handler
        def handle_compare_btn_click(query: str, docs_for_comparison: List[str], stream_compare: bool, progress=gr.Progress()):
            if not query.strip():
                yield "Please enter a question for comparison."
                return
            if len(docs_for_comparison) < 2:
                yield "Please select at least 2 documents to compare."
                return

            if stream_compare and config.enable_streaming:
                streaming_handler = StreamingHandler()
                thread = threading.Thread(target=app.comparison_engine.compare_documents_streaming,
                                          args=(query, app.doc_manager, docs_for_comparison, streaming_handler))
                thread.start()
                current_comparison = ""
                for chunk in streaming_handler.get_stream():
                    current_comparison = chunk
                    yield current_comparison
                thread.join()
            else:
                comparison_result = app.compare_documents(query, docs_for_comparison, progress=progress)
                yield comparison_result

        compare_btn.click(handle_compare_btn_click,
                          [compare_query_box, compare_doc_selector, stream_compare_cb],
                          [comparison_output_md])

        # Settings handlers
        enable_diagnostics_cb.change(lambda x: setattr(app.diagnostics_logger, 'enabled', x), [enable_diagnostics_cb], None)

        def update_vlm_config_options(mode):
            app.config.skip_vlm = (mode == "skip")
            app.config.vlm_text_only_mode = (mode == "text_only")
            if app.config.skip_vlm and app.vlm_handler and app.vlm_handler.model:
                print("VLM mode changed to skip. Unloading VLM.")
                app.vlm_handler.unload()
            return f"VLM mode set to {mode}"

        vlm_mode_radio.change(update_vlm_config_options, [vlm_mode_radio], [upload_status])

        if torch.cuda.is_available():
            def trigger_gpu_status_update():
                return get_gpu_status_str()

            # FIXED: Updated for Gradio 5 compatibility
            gpu_timer = gr.Timer(5) # Timer interval of 5 seconds
            gpu_timer.tick(trigger_gpu_status_update, inputs=None, outputs=[gpu_status])

            def handle_clear_gpu_btn_click():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return get_gpu_status_str()
            clear_gpu_btn.click(handle_clear_gpu_btn_click, [], [gpu_status])

    return demo

# ===========================================
# MAIN EXECUTION
# ===========================================

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print("🚀 Starting FIXED Multi-PDF RAG System v9.3...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"FAISS Available: {FAISS_AVAILABLE}")
    print("📊 Embedding Model: NovaSearch/stella_en_400M_v5")
    print("🌙 VLM Model: Moondream 2B (4-bit quantized)")
    print("🤖 LLM: Dynamic selection with support for Gemma 3 4B and Qwen 3 8B")
    print("✨ Features: Map-Reduce Summarization, Multi-File Upload, Streaming Output, Diagnostics")
    print("🧠 Thinking Mode: Available for Qwen models with automatic tag extraction")
    print("🔧 DEBUG MODE: Type 'debug: your question' in chat or use the debug panel")
    print("")
    print("🔧 MAJOR FIXES APPLIED:")
    print("   • FIXED: FAISS L2 distance score interpretation (lower = better)")
    print("   • FIXED: Document filtering logic that was removing relevant documents")
    print("   • FIXED: Improved chunking strategy for better context preservation")
    print("   • FIXED: Better score normalization and thresholds")
    print("   • FIXED: Enhanced debug output with proper score understanding")
    print("")

    gradio_app = create_gradio_interface()
    gradio_app.queue().launch(
        server_name="0.0.0.0", server_port=8855, share=True, show_error=True
    )