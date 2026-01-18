ENV_NAME := canine-gpu
CUDA_TORCH_URL := https://download.pytorch.org/whl/cu121

.PHONY: setup clean

setup:
	. $$(conda info --base)/etc/profile.d/conda.sh && \
	conda create -y -n $(ENV_NAME) python=3.10 && \
	conda activate $(ENV_NAME) && \
	pip install --upgrade pip && \
	pip install torch torchvision torchaudio --index-url $(CUDA_TORCH_URL) && \
	pip install transformers sentencepiece accelerate pandas numpy scikit-learn matplotlib tqdm emoji ipykernel && \
	python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))" && \
	python -c "import torch; print('✅ Setup complete'); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

clean:
	conda remove -y -n $(ENV_NAME) --all
