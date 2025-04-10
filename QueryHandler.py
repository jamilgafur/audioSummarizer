import re
import torch
from transformers import pipeline
from ollama import chat
from typing import Optional
import os


class TextProcessor:
    def __init__(self, huggingface_model_name: str, ollama_model_name: str, temperature: float = 1.0, device: Optional[str] = None):
        print(f"Initializing TextProcessor with HuggingFace model: {huggingface_model_name} and Ollama model: {ollama_model_name}")
        self.huggingface_model_name = huggingface_model_name
        self.ollama_model_name = ollama_model_name
        self.temperature = temperature
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print("Initializing HuggingFace pipeline for text generation.")
        self.huggingface_pipeline = pipeline("text-generation", model=self.huggingface_model_name, device=self.device)

    def clean_text(self, raw_text: str, sys_prompt: str = "grammar:", chunk_size: int = 2000) -> str:
        print(f"Cleaning raw text with chunk size: {chunk_size}")
        paragraphs = [p for p in raw_text.strip().split('\n') if p.strip()]
        if len(paragraphs) > 2:
            raw_text = '\n'.join(paragraphs[1:-1])
        chunks = self._split_text_into_chunks(raw_text, chunk_size)
        print(f"Text split into {len(chunks)} chunks.")
        cleaned_chunks = [self._process_chunk(chunk, sys_prompt) for chunk in chunks]
        cleaned_text = "\n".join(cleaned_chunks)
        print("Text cleaning completed.")
        return cleaned_text

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> list:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _remove_duplicates(self, input_string: str) -> str:
        words = input_string.split()
        seen = set()
        unique_words = [word for word in words if word.lower() not in seen and not seen.add(word.lower())]
        return ' '.join(unique_words)

    def _process_chunk(self, chunk: str, sys_prompt: str) -> str:
        cleaned_chunk = self._basic_cleaning(chunk)
        cleaned_chunk = self._call_huggingface_model(cleaned_chunk, sys_prompt)
        return cleaned_chunk

    def _call_huggingface_model(self, text: str, sys_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        print(f"Calling HuggingFace model with text of length {len(text)} and system prompt.")
        inputs = self.huggingface_pipeline.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        try:
            output = self.huggingface_pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens)
            cleaned_text = self.huggingface_pipeline.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during HuggingFace model generation: {e}")
            return text

        return self._remove_duplicates(cleaned_text)

    def _basic_cleaning(self, text: str) -> str:
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[.*?\]', '', text)
        return text

    def generate_script(self, input_text: str, sys_prompt: str = """Update and summarize the provided text into a concise, clear, and well-structured report. The tone should be professional, yet approachable, suitable for a single speaker presenting the information. Break down longer sentences into shorter, clearer ones, aiming for clarity and smooth flow of ideas. Each sentence should be around 50 words to maintain a steady pace when read aloud. Ensure the information is presented logically, with appropriate transitions between topics. Make the report easy to follow, with emphasis on key points, and avoid overly complex or technical jargon unless necessary. Keep the language formal and polished, suitable for a professional presentation. Do not include any references to the original source text.""", save_path: Optional[str] = "podcast_script.txt", chunks: int = 20) -> str:
        paragraphs = [p for p in input_text.strip().split('\n') if p.strip()]
        if len(paragraphs) > 2:
            input_text = '\n'.join(paragraphs[1:-1])

        parts = [input_text[i:i + len(input_text) // chunks] for i in range(0, len(input_text), len(input_text) // chunks)]
        final_output = ""

        for i, part in enumerate(parts):
            if part.strip():
                title_prompt = f"Give a short, engaging title for this episode based on the following content: {part[:400]}"
                title_response = chat(messages=[{'role': 'user', 'content': title_prompt}], model=self.ollama_model_name)
                title = title_response['message']['content'].strip()
                response = chat(messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': part}], model=self.ollama_model_name)
                script = response['message']['content']
                final_output += f"\n\n=== {title} ===\n{script}\n"
                print(f"\n[Episode {i + 1}: {title}]\n{script}\n")

        if save_path:
            with open(save_path, 'w') as file:
                file.write(final_output)

        return final_output
