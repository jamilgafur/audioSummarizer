import re
import torch
from transformers import pipeline
from pydantic import BaseModel, validator, root_validator
from typing import List, Tuple
from ollama import chat
import os


class TextProcessor:
    def __init__(self, huggingface_model_name, ollama_model_name, temperature=1.0, device=None):
        print(f"Initializing TextProcessor with HuggingFace model: {huggingface_model_name} and Ollama model: {ollama_model_name}")
        self.huggingface_model_name = huggingface_model_name
        self.ollama_model_name = ollama_model_name
        self.temperature = temperature
        
        # Set device: 'cuda' if GPU is available, otherwise 'cpu'
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize Hugging Face pipeline for text generation
        print("Initializing HuggingFace pipeline for text generation.")
        self.huggingface_pipeline = pipeline("text-generation", model=self.huggingface_model_name, device=self.device)

    def clean_text(self, raw_text, sys_prompt, chunk_size=2000):
        print(f"Cleaning raw text with chunk size: {chunk_size}")
        
        # Step 1: Split the raw text into smaller chunks to avoid long token sequences.
        chunks = self._split_text_into_chunks(raw_text)
        print(f"Text split into {len(chunks)} chunks.")
        
        # Step 2: Clean the chunks sequentially.
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            cleaned_chunk = self._process_chunk(chunk, sys_prompt)
            cleaned_chunks.append(cleaned_chunk)
            torch.cuda.empty_cache()  # Clear GPU memory after each chunk processing

        # Step 3: Combine the cleaned chunks back into one text.
        cleaned_text = "\n".join(cleaned_chunks)
        print("Text cleaning completed.")
        return cleaned_text

    def _remove_duplicates(self, input_string):
        """
        Removes duplicate words or phrases from the input string.
        The method will handle cases like repeated words in a sequence.
        """
        # Split the input string into words (or phrases if needed)
        words = input_string.split()

        # Use a set to remove duplicates while preserving order
        seen = set()
        unique_words = []
        
        for word in words:
            # Check if the word has already been encountered
            if word.lower() not in seen:
                unique_words.append(word)
                seen.add(word.lower())  # case-insensitive check

        # Join the words back into a single string
        cleaned_string = ' '.join(unique_words)
        
        return cleaned_string

    def _process_chunk(self, chunk, sys_prompt):
        print(f"Processing chunk of length {len(chunk)}.")
        # Clean the chunk
        cleaned_chunk = self._basic_cleaning(chunk)
        print("Basic cleaning completed.")
        # Call the Hugging Face model for further cleaning if needed
        cleaned_chunk = self._call_huggingface_model(cleaned_chunk, 'grammar: ')
        print("Chunk processed by Hugging Face model.")
        print(f"{cleaned_chunk}")
        
        return cleaned_chunk

    def _call_huggingface_model(self, text, sys_prompt, max_new_tokens=None):
        print(f"Calling Hugging Face model with text length: {len(text)} and system prompt.")

        # Concatenate sys_prompt with text to form the complete input

        # Tokenize the input text
        try:
            inputs = self.huggingface_pipeline.tokenizer(text, return_tensors="pt", truncation=True)
            print(f"Tokenization completed. Input tokens length: {inputs.input_ids.shape[1]}")
        except Exception as e:
            print(f"Error during tokenization: {e}")
            raise

        # Move inputs to the same device as the model (either CPU or GPU)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        print(f"Inputs moved to device: {self.device}")

        try:
            # Generate text using the Hugging Face model
            output = self.huggingface_pipeline.model.generate(
                **inputs
            )
            print("Text generation completed.")
        except Exception as e:
            print(f"Error during text generation: {e}")
            raise

        # Decode the output tokens into text
        cleaned_text = self.huggingface_pipeline.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated text preview: {cleaned_text[:100]}...")  # Show a preview of the generated text
        
        # Clear GPU memory after processing
        torch.cuda.empty_cache()
        
        return self._remove_duplicates(cleaned_text)

    def _basic_cleaning(self, text):
        print(f"Performing basic cleaning on text of length {len(text)}.")
        # Perform basic text cleaning steps (example: removing LaTeX, extra spaces, etc.)
        text = re.sub(r'\$.*?\$', '', text)  # Remove LaTeX math
        text = re.sub(r'\n+', ' ', text)  # Remove extra newlines
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = re.sub(r'\[.*?\]', '', text)  # Remove footnotes or references
        print("Basic cleaning completed.")
        return text
    
    def generate_script(self, input_text, sys_prompt, save_path="podcast_script.txt", chunks=20):
        """
        Generate a podcast script using the provided input and system prompt.
        This method will interact with Ollama's chat function to generate a conversational podcast script.
        
        Args:
            input_text (str): The content or outline to be used for generating the script.
            sys_prompt (str): The system-level prompt that guides the tone and structure of the podcast.
            save_path (str): The path to save the generated script.
            
        Returns:
            str: The generated podcast script.
        """
        
        # Split the input text into 5 roughly equal parts
        parts = [input_text[i:i + len(input_text)//chunks] for i in range(0, len(input_text), len(input_text)//chunks)]
        
        # Initialize a variable to store the final merged output
        final_output = ""
        
        # Initialize context that will be passed along through each part
        context = sys_prompt  # Start with the initial system prompt as context
        
        # Process each part individually and generate the script
        for i, part in enumerate(parts):
            # Skip any empty lines
            if part.strip():
                # Append the current part to the context for continuity
                context += f"\nUser: {part}"
                
                # Make the request using Ollama's chat function for conversational generation
                response = chat(
                    messages=[
                        {'role': 'system', 'content': context},
                        {'role': 'user', 'content': part}
                    ],
                    model=self.ollama_model_name,
                )
                
                # Access the message content and append it to the final output
                final_output += response['message']['content'] + "\n"
                
                # If it's not the last part, generate an intermission summary and introduction for the next person
                if i < len(parts) - 1:
                    # Request an intermission summary and introduction for the next speaker
                    intermission_query = "Summarize what was just said and introduce a new person who will be going over the next part."
                    intermission_response = chat(
                        messages=[
                            {'role': 'system', 'content': sys_prompt},
                            {'role': 'user', 'content': intermission_query}
                        ],
                        model=self.ollama_model_name,
                    )
                    # Append the intermission content to the final output
                    final_output += "\n" + "INTERMISSION_UPDATE_LOGGER_WAIT" + intermission_response['message']['content'] + "\n"
                
                # Update context with the latest response for the next iteration
                context += f"\nAssistant: {response['message']['content']}"
        
        # Optionally save the result to a file if save_path is provided
        if save_path:
            with open(save_path, 'w') as file:
                file.write(final_output)
        
        # Return the final merged podcast script
        return final_output



    def _split_text_into_chunks(self, text):
        # Split text into smaller chunks to avoid long token sequences
        return text.split('\n')
