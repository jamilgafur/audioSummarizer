import re
import torch
from transformers import pipeline
from ollama import chat
from typing import Optional
import os
import nltk

nltk.download('punkt')

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
        """
        Clean and process the raw text by splitting it into manageable chunks.
        """
        print(f"Cleaning raw text with chunk size: {chunk_size}")
        paragraphs = [p for p in raw_text.strip().split('\n') if p.strip()]
        if len(paragraphs) > 2:
            # Discard the first and last paragraph if there are many paragraphs.
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
        print(f"Calling HuggingFace model with text length {len(text)} and system prompt.")
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

    def generate_script(
        self, 
        input_text: str, 
        sys_prompt: Optional[str] = (
            "You are a professional teleprompt scriptwriter with expertise in transforming complex content into an engaging, clear, and well-organized script for a high school audience. "
            "Your job is to rewrite the following text into a teleprompter script by simplifying topics, breaking long paragraphs into digestible pieces, and organizing the information logically. "
            "The script should start with a concise, captivating title, followed by a clear, conversational narrative. "
            "Use bullet points or numbered lists if it helps clarify the key points. "
            "Avoid any commentary about the original text. "
            "Now, use the following content to create the script:"
        ), 
        save_path: Optional[str] = "podcast_script.txt", 
        chunks: int = 20
    ) -> str:
        """
        Generate a teleprompter script by splitting the input text into manageable parts,
        generating a title for each part, and transforming it into an engaging script using Ollama.
        """
        # Split input text into sentences for smarter grouping.
        sentences = nltk.sent_tokenize(input_text)
        num_sentences = len(sentences)
        group_size = max(1, num_sentences // chunks)
        parts = [' '.join(sentences[i:i+group_size]) for i in range(0, num_sentences, group_size)]
        
        final_output = []

        for i, part in enumerate(parts):
            if part.strip():
                # Generate a title using a simple prompt.
                title_prompt = f"You are clickbait youtuber. Generate a concise and engaging title with 10 words or less for the following content. MAKE SURE IT IS 10 WORDS OR LESS.:\n\n {part[:100]}"
                title_response = chat(
                    messages=[{'role': 'user', 'content': title_prompt}], 
                    model=self.ollama_model_name
                )
                title = title_response['message']['content'].strip()

                # Generate the teleprompter script using the detailed system prompt.
                response = chat(
                    messages=[
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': part}
                    ],
                    model=self.ollama_model_name
                )
                script = response['message']['content'].strip()
                episode_output = f"\n\nEpisode {i}: {title} \n{script}\n"
                final_output.append(episode_output)
                print(f"\n[Episode {i + 1}: {title}]\n{script}\n")

        if save_path:
            # Save each generated script part to a separate text file.
            for i, text in enumerate(final_output):
                file_base, ext = os.path.splitext(save_path)
                file_path = f"{file_base}_part_{i + 1}{ext if ext else '.txt'}"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    print(f"Saved: {file_path}")

        return "\n".join(final_output)
