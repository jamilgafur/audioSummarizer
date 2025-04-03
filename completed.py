import os
import argparse
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from transformers import AutoProcessor, BarkModel
import numpy as np
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import pdfplumber
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import warnings
import gc
from transformers import AutoProcessor, AutoModelForTextToWaveform

warnings.filterwarnings('ignore')
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")

# Define system prompts for different steps in the pipeline
SYS_PROMPT_CLEANING = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.
Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive
Please be smart with what you remove and be creative ok?
Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED
Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES
ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

SYS_PROMPT_PODCAST = """
You are the a world-class podcast writer, you have worked as a ghost writer for many famous podcasters
We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.
You have won multiple podcast awards for your writing.
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic.
Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker.
It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait
ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""

SYSTEM_PROMPT_REWRITE = """
You are an international oscar winning screenwriter
You have been working with multiple award winning podcasters.
Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.
Make it as engaging as possible, 
REMEMBER THIS WITH YOUR HEART
Use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS
It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait
Please re-write to make it as characteristic as possible

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE
"""

class PodcastGenerator:
    def __init__(self, model_name, max_tokens=8126, temperature=1.0):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def read_pdf(self, file_path: str, max_chars: int = 100000):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("File is not a PDF")
        
        extracted_text = []
        total_chars = 0
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    if total_chars + len(text) > max_chars:
                        extracted_text.append(text[: max_chars - total_chars])
                        break
                    extracted_text.append(text)
                    total_chars += len(text)
        return '\n'.join(extracted_text)

    def clean_text(self, text, sys_prompt, save_path="cleaned_text.txt"):
        # Check if file exists
        if os.path.exists(save_path):
            with open(save_path, "r") as file:
                cleaned_text = file.read()
            print(f"Loaded cleaned text from {save_path}.")
        else:
            conversation = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ]
            
            conversation_text = "\n".join([msg["content"] for msg in conversation])
            
            inputs = self.pipeline.tokenizer(conversation_text, return_tensors="pt").to(self.pipeline.device)
            max_token_length = self.pipeline.model.config.max_position_embeddings
            
            # Ensure the token length does not exceed the model's maximum length
            if inputs.input_ids.shape[1] > max_token_length:
                inputs = {key: value[:, :max_token_length] for key, value in inputs.items()}
            
            with torch.no_grad():
                output = self.pipeline.model.generate(**inputs, temperature=self.temperature, top_p=0.9, max_new_tokens=512)
            
            cleaned_text = self.pipeline.tokenizer.decode(output[0], skip_special_tokens=True)[len(inputs['input_ids']):].strip()
            
            # Save cleaned text
            with open(save_path, "w") as file:
                file.write(cleaned_text)
            print(f"Saved cleaned text to {save_path}.")
        
        return cleaned_text

    def generate_script(self, input_text, sys_prompt, save_path="podcast_script.txt"):
        # Check if file exists
        if os.path.exists(save_path):
            with open(save_path, "r") as file:
                podcast_script = file.read()
            print(f"Loaded podcast script from {save_path}.")
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": input_text},
            ]
            outputs = self.pipeline(messages, max_new_tokens=self.max_tokens, temperature=self.temperature)
            podcast_script = outputs[0]["generated_text"][-1]['content']
            
            # Save podcast script
            with open(save_path, "w") as file:
                file.write(podcast_script)
            print(f"Saved podcast script to {save_path}.")
        
        return podcast_script

def save_audio_to_wav(audio_data, rate, output_filename):
    audio_data = np.clip(audio_data, -1, 1)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(output_filename, rate, audio_data_int16)
    print(f"Audio saved as {output_filename}")

def generate_audio_with_bark(text, sentences_per_chunk=3, save_path="output_audio"):
    """
    Generates audio for larger chunks (grouped by sentences) of the provided text using Suno Bark TTS,
    saves them as separate .wav files, and merges them into one final audio file.

    Args:
    - text (str): The text to generate audio from.
    - sentences_per_chunk (int): The number of sentences in each chunk. Defaults to 5.
    - save_path (str): The path to save the final generated audio file.
    """
    # Voice preset (you can modify this for different voices)
    voice_preset = "v2/en_speaker_6"  # Example voice preset

    # Split the text into sentences (by splitting on periods followed by spaces)
    sentences = text.split('.')
    
    # Create chunks based on the number of sentences per chunk
    chunks = [sentences[i:i + sentences_per_chunk] for i in range(0, len(sentences), sentences_per_chunk)]

    audio_files = []

    for idx, chunk in enumerate(chunks):
        chunk_text = ' '.join(chunk).strip()  # Join sentences in the chunk to form a larger text block
        print(chunk_text)
        if chunk_text:
            # Use the processor to prepare inputs for the model
            inputs = processor(chunk_text, voice_preset=voice_preset)

            # Generate audio using the Bark model
            audio_array = model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()

            # Save audio to a temporary file for this chunk
            chunk_filename = f"{save_path}_chunk_{idx + 1}.wav"
            write_wav(chunk_filename, 22050, audio_array)  # Assuming Bark uses 22050 Hz sample rate
            audio_files.append(chunk_filename)
            print(f"Generated audio for chunk {idx + 1}: {chunk_filename}")

    # Merge the generated audio files into one large file
    merged_audio = AudioSegment.empty()  # Start with an empty audio segment
    for audio_file in audio_files:
        chunk_audio = AudioSegment.from_wav(audio_file)
        merged_audio += chunk_audio  # Append each chunk's audio

    # Save the final merged audio file
    merged_filename = f"{save_path}_merged_podcast.wav"
    merged_audio.export(merged_filename, format="wav")
    print(f"Final merged audio saved as {merged_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate podcast from PDF using Hugging Face models.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model to use from Hugging Face")
    parser.add_argument('--max_tokens', type=int, default=8126, help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    generator = PodcastGenerator(model_name=args.model, max_tokens=args.max_tokens, temperature=args.temperature)

    # Check if the cleaned text file exists, else process and save it
    cleaned_text_path = "cleaned_text.txt"
    if os.path.exists(cleaned_text_path):
        with open(cleaned_text_path, "r") as file:
            cleaned_text = file.read()
        print(f"Loaded cleaned text from {cleaned_text_path}.")
    else:
        print("Cleaning text...")
        pdf_text = generator.read_pdf(args.pdf_path)  # Ensure PDF is read if not already loaded
        cleaned_text = generator.clean_text(pdf_text, SYS_PROMPT_CLEANING, save_path=cleaned_text_path)

    # Check if the podcast script file exists, else generate and save it
    podcast_script_path = "podcast_script.txt"
    if os.path.exists(podcast_script_path):
        with open(podcast_script_path, "r") as file:
            podcast_script = file.read()
        print(f"Loaded podcast script from {podcast_script_path}.")
    else:
        print("Generating podcast script...")
        podcast_script = generator.generate_script(cleaned_text, SYS_PROMPT_PODCAST, save_path=podcast_script_path)

    # Check if the rewritten podcast script file exists, else generate and save it
    rewritten_script_path = "rewritten_script.txt"
    if os.path.exists(rewritten_script_path):
        with open(rewritten_script_path, "r") as file:
            rewritten_script = file.read()
        print(f"Loaded rewritten script from {rewritten_script_path}.")
    else:
        print("Rewriting podcast script...")
        rewritten_script = generator.generate_script(podcast_script, SYSTEM_PROMPT_REWRITE, save_path=rewritten_script_path)
    
    clear_gpu_memory()
    print("Generating TTS audio with Bark...")
    generate_audio_with_bark(rewritten_script, save_path=args.pdf_path)


def clear_gpu_memory():
    """Clears unused GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared.")
    else:
        print("CUDA is not available. No memory cleared.")

if __name__ == "__main__":
    main()
