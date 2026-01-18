ENV_NAME := canine-gpu
CUDA_TORCH_URL := https://download.pytorch.org/whl/cu121

.PHONY: setup clean

setup:
	conda create -y -n $(ENV_NAME) python=3.10 && \
	. $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	pip install --upgrade pip && \
	pip install torch torchvision torchaudio --index-url $(CUDA_TORCH_URL) && \
	pip install \
		transformers \
		sentencepiece \
		accelerate \
		pandas \
		numpy \
		scikit-learn \
		matplotlib \
		tqdm \
		emoji \
		ipykernel && \
	python -m ipykernel install --user \
		--name $(ENV_NAME) \
		--display-name "Python ($(ENV_NAME))" && \
	python - << 'EOF'
import torch
print("✅ Setup complete")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

clean:
	conda remove -y -n $(ENV_NAME) --all
