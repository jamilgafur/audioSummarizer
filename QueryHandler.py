import re
import torch
from transformers import pipeline
from ollama import chat
from typing import Optional
import os


class TextProcessor:
    def __init__(self, huggingface_model_name: str, ollama_model_name: str, temperature: float = 1.0, device: Optional[str] = None):
        """
        Initializes the TextProcessor with specified HuggingFace and Ollama models.
        
        Args:
            huggingface_model_name: Name of the HuggingFace model to be used.
            ollama_model_name: Name of the Ollama model to be used.
            temperature: Temperature for text generation (used with HuggingFace).
            device: The device ('cpu' or 'cuda') to run the models on.
        """
        print(f"Initializing TextProcessor with HuggingFace model: {huggingface_model_name} and Ollama model: {ollama_model_name}")
        self.huggingface_model_name = huggingface_model_name
        self.ollama_model_name = ollama_model_name
        self.temperature = temperature
        
        # Set device: 'cuda' if GPU is available, otherwise 'cpu'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Hugging Face pipeline for text generation
        print("Initializing HuggingFace pipeline for text generation.")
        self.huggingface_pipeline = pipeline("text-generation", model=self.huggingface_model_name, device=self.device)

    def clean_text(self, raw_text: str, sys_prompt: str = "grammar:", chunk_size: int = 2000) -> str:
        """
        Clean and process raw input text by splitting into chunks, processing them individually, and then recombining.

        Args:
            raw_text: The raw input text to be cleaned.
            sys_prompt: A system-level prompt for text generation (default is grammar-related cleaning).
            chunk_size: The maximum size of each chunk of text to be processed.
        
        Returns:
            The cleaned and processed text.
        """
        print(f"Cleaning raw text with chunk size: {chunk_size}")
        
        # Step 1: Split the raw text into smaller chunks
        chunks = self._split_text_into_chunks(raw_text, chunk_size)
        print(f"Text split into {len(chunks)} chunks.")
        
        # Step 2: Process each chunk
        cleaned_chunks = [self._process_chunk(chunk, sys_prompt) for chunk in chunks]

        # Step 3: Combine the cleaned chunks
        cleaned_text = "\n".join(cleaned_chunks)
        print("Text cleaning completed.")
        return cleaned_text

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> list:
        """
        Split the input text into smaller chunks based on the chunk size.

        Args:
            text: The raw input text to be split.
            chunk_size: The maximum size of each chunk.

        Returns:
            A list of text chunks.
        """
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _remove_duplicates(self, input_string: str) -> str:
        """
        Removes duplicate words or phrases from the input string.

        Args:
            input_string: The string from which duplicates need to be removed.

        Returns:
            The string without duplicates.
        """
        words = input_string.split()
        seen = set()
        unique_words = [word for word in words if word.lower() not in seen and not seen.add(word.lower())]
        return ' '.join(unique_words)

    def _process_chunk(self, chunk: str, sys_prompt: str) -> str:
        """
        Process a single chunk of text by cleaning it and running it through the HuggingFace model.

        Args:
            chunk: The chunk of text to be processed.
            sys_prompt: A system-level prompt for the HuggingFace model.

        Returns:
            The processed chunk of text.
        """
        cleaned_chunk = self._basic_cleaning(chunk)
        cleaned_chunk = self._call_huggingface_model(cleaned_chunk, sys_prompt)
        return cleaned_chunk

    def _call_huggingface_model(self, text: str, sys_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Call the HuggingFace model for text generation.

        Args:
            text: The input text to be processed.
            sys_prompt: The system-level prompt to guide the model's generation.
            max_new_tokens: Maximum number of new tokens to generate (optional).

        Returns:
            The model's generated output text.
        """
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
        """
        Perform basic cleaning on the input text by removing unwanted characters and patterns.

        Args:
            text: The input text to be cleaned.

        Returns:
            The cleaned text.
        """
        text = re.sub(r'\$.*?\$', '', text)  # Remove LaTeX math
        text = re.sub(r'\n+', ' ', text)  # Remove extra newlines
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = re.sub(r'\[.*?\]', '', text)  # Remove footnotes or references
        return text

    def generate_script(self, input_text: str, sys_prompt: str = """Update and summarize the provided text into a concise, clear, and well-structured report. The tone should be professional, yet approachable, suitable for a single speaker presenting the information. Break down longer sentences into shorter, clearer ones, aiming for clarity and smooth flow of ideas. Each sentence should be around 50 words to maintain a steady pace when read aloud. Ensure the information is presented logically, with appropriate transitions between topics. Make the report easy to follow, with emphasis on key points, and avoid overly complex or technical jargon unless necessary. Keep the language formal and polished, suitable for a professional presentation. Do not include any references to the original source text.""", save_path: Optional[str] = "podcast_script.txt", chunks: int = 20) -> str:        
        """
        Generate a podcast script by interacting with Ollama's chat function.

        Args:
            input_text: The raw input text to be transformed into a script.
            sys_prompt: The system-level prompt for generating a fun and engaging podcast.
            save_path: Path to save the generated script.
            chunks: Number of chunks to split the input text into for processing.

        Returns:
            The generated podcast script.
        """
        parts = [input_text[i:i + len(input_text) // chunks] for i in range(0, len(input_text), len(input_text) // chunks)]
        
        final_output = ""

        for i, part in enumerate(parts):
            if part.strip():
                # Process each chunk independently without referring to previous context
                response = chat(messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': part}],
                                model=self.ollama_model_name)
                print(response['message']['content'])
                final_output += response['message']['content'] + "\n"

        if save_path:
            with open(save_path, 'w') as file:
                file.write(final_output)
        
        return final_output
