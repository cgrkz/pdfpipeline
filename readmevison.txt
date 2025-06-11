# 🧠 Qwen2.5-VL Inference App

This project sets up a visual-language inference pipeline using [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — an advanced multimodal model from Alibaba's Qwen team. It allows image + text input and generates language-rich outputs.

---

## 🚀 Features

- Uses `Qwen2_5_VLForConditionalGeneration` and processor from HuggingFace
- GPU-accelerated with PyTorch
- Works with `transformers` from source
- Compatible with HuggingFace's `accelerate` and `qwen-vl-utils`

---

## 🛠️ Setup Instructions

### ✅ Prerequisites

- Python >= 3.8
- CUDA-compatible GPU (recommended)
- Git
- pip

### 📦 Install Dependencies

```bash
# Step 1: Uninstall any existing transformers version
pip uninstall transformers -y

# Step 2: Install transformers from GitHub source (required for Qwen2.5-VL classes)
pip install git+https://github.com/huggingface/transformers

# Step 3: Install supporting dependencies
pip install torch==2.6.0
pip install accelerate==1.4.0
pip install safetensors==0.5.2

# Step 4: Install Qwen VL utils
pip install qwen-vl-utils[decord]==0.0.8
