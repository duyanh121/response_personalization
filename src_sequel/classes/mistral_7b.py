import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MistralGenerator:
    def __init__(
        self,
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda",
        dtype=torch.float16,
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True
        )

        # Important for Mistral
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto"
        )

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
