# =========================
# Configuration
# =========================
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# =========================
# Targets
# =========================

.PHONY: help setup venv install install-torch clean check

help:
	@echo "Available targets:"
	@echo "  make setup        -> Full environment setup (recommended)"
	@echo "  make venv         -> Create virtual environment"
	@echo "  make install      -> Install all Python dependencies"
	@echo "  make install-torch-> Install PyTorch (CUDA if available)"
	@echo "  make check        -> Verify imports"
	@echo "  make clean        -> Remove virtual environment"

# =========================
# Setup
# =========================

setup: venv install-torch install check

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

# =========================
# PyTorch (CUDA-aware)
# =========================

install-torch:
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
	$(PIP) install torch torchvision torchaudio

# =========================
# Python dependencies
# =========================

install:
	$(PIP) install \
		transformers \
		sentencepiece \
		accelerate \
		pandas \
		numpy \
		scikit-learn \
		matplotlib \
		tqdm \
		emoji

# =========================
# Sanity check
# =========================

check:
	$(PY) - << 'EOF'
import torch
import transformers
import pandas
import numpy
import sklearn
import matplotlib
import tqdm
import emoji

print("✅ All core dependencies imported successfully")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# =========================
# Cleanup
# =========================

clean:
	rm -rf $(VENV)
