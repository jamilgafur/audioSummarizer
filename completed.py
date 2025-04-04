from FileManager import FileManager
from QueryHandler import TextProcessor  # Updated import
from AudioGenerator import AudioGenerator
import argparse
import os
# Define system prompts for different steps in the pipeline
SYS_PROMPT_CLEANING = """grammar:"""

SYS_PROMPT_PODCAST = """
You are an international oscar winnning screenwriter. You have been working with multiple award winning podcasters. Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind. Make it as engaging as possible, Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the podcast.Be super fun and engaging and keep it really catchy and almost borderline click bait. Please re-write to make it as characteristic as possible.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate podcast from PDF using Hugging Face models.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument('--modelOllama', type=str, default="llama3.2", help="Model to use from Hugging Face")
    parser.add_argument('--modelHugging', type=str, default="facebook/opt-125m", help="Model to use from Hugging Face")
    parser.add_argument('--max_tokens', type=int, default=8126, help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")
    parser.add_argument('--pdf_text_file', type=str, default="pdf_text.txt", help="Sampling temperature")
    parser.add_argument('--cleaned_text_file', type=str, default="cleaned.txt", help="Sampling temperature")
    parser.add_argument('--audiomodelname', type=str,default="suno/bark-small", help="Sampling temperature")

    args = parser.parse_args()

    # Instantiate classes
    file_manager = FileManager()
    text_processor = TextProcessor(huggingface_model_name=args.modelHugging, ollama_model_name=args.modelOllama, temperature=args.temperature)

    # Read PDF and clean text

    # Check if PDF text file exists
    if os.path.exists(args.pdf_text_file):
        print(f"Loading existing PDF text from {args.pdf_text_file}")
        with open(args.pdf_text_file, 'r') as file:
            pdf_text = file.read()
    else:
        # Read PDF and save extracted text
        pdf_text = file_manager.read_pdf(args.pdf_path)
        with open(args.pdf_text_file, 'w') as file:
            file.write(pdf_text)
        print(f"Extracted PDF text saved to {args.pdf_text_file}")

    if os.path.exists("podcast_script.txt"):
        print(f"Loading existing podcast script from podcast_script.txt")
        with open("podcast_script.txt", 'r') as file:
            podcast_script = file.read()
    else:
        # Generate podcast script
        podcast_script = text_processor.generate_script(pdf_text, SYS_PROMPT_PODCAST)

    file_manager.clear_gpu_memory()

    audiogen = AudioGenerator()
    audiogen.generate_audio_from_text(podcast_script)
    # Clear GPU memory
    file_manager.clear_gpu_memory()



if __name__ == "__main__":
    main()
