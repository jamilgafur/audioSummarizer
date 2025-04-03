import os
import argparse
import torch
import pdfplumber
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle
import warnings
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from pydub import AudioSegment
import io
import numpy as np
from scipy.io import wavfile

warnings.filterwarnings('ignore')

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

    def clean_text(self, text, sys_prompt):
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
        
        return self.pipeline.tokenizer.decode(output[0], skip_special_tokens=True)[len(inputs['input_ids']):].strip()

    def generate_script(self, input_text, sys_prompt):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": input_text},
        ]
        outputs = self.pipeline(messages, max_new_tokens=self.max_tokens, temperature=self.temperature)
        return outputs[0]["generated_text"][-1]['content']

    def generate_audio(self, text, model, tokenizer):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.pipeline.device)
        generation = model.generate(input_ids=input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        return audio_arr, model.config.sampling_rate



def setup_parler_tts():
    model_name = "parler-tts/parler-tts-mini-v1"
    torch_device = "cuda:0"  # or "mps" for Mac
    torch_dtype = torch.bfloat16
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="eager"
    ).to(torch_device, dtype=torch_dtype)
    compile_mode = "default"
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode=compile_mode)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def numpy_to_audio_segment(audio_data, rate):
    device = "cpu"
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.to(device)
    
    audio_arr = audio_data.cpu().numpy() if isinstance(audio_data, torch.Tensor) else np.array(audio_data, dtype=np.float32)
    audio_arr = np.interp(audio_arr, (audio_arr.min(), audio_arr.max()), (-1, 1))
    
    audio_segment = AudioSegment(
        audio_arr.tobytes(),
        frame_rate=rate,
        sample_width=audio_arr.itemsize,
        channels=1
    )
    return audio_segment


def save_audio_to_wav(audio_data, rate, output_filename):
    audio_data = np.clip(audio_data, -1, 1)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(output_filename, rate, audio_data_int16)
    print(f"Audio saved as {output_filename}")


import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread

def generate_with_streaming(text, description, model, tokenizer, chunk_size_in_s=0.5, save_path=""):
    frame_rate = model.audio_encoder.config.frame_rate
    sampling_rate = model.audio_encoder.config.sampling_rate
    
    # Create the streamer
    play_steps = int(frame_rate * chunk_size_in_s)
    streamer = ParlerTTSStreamer(model, device="cuda:0", play_steps=play_steps)

    # Tokenize inputs
    inputs = tokenizer(description, return_tensors="pt").to("cuda:0")
    prompt = tokenizer(text, return_tensors="pt").to("cuda:0")

    # Set up the generation arguments
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )
    
    # Function to handle the generation in a separate thread
    def generate_audio():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        thread.join()  # Wait for the thread to finish

    # Start the generation thread
    generate_audio()

    # Iterate over chunks of audio
    audio_data_all = []
    for new_audio in streamer:
        if new_audio.shape[0] == 0:
            break
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 4)} seconds")
        audio_data_all.append(new_audio)

    # Concatenate all audio chunks and save to a file
    if audio_data_all:
        audio_data_all = np.concatenate(audio_data_all, axis=0)
        audio_filename = f"{save_path}_podcast.wav"
        save_audio_to_wav(audio_data_all, sampling_rate, audio_filename)
    



def main():
    parser = argparse.ArgumentParser(description="Generate podcast from PDF using Hugging Face models.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model to use from Hugging Face")
    parser.add_argument('--max_tokens', type=int, default=8126, help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    generator = PodcastGenerator(model_name=args.model, max_tokens=args.max_tokens, temperature=args.temperature)

    print("Reading and processing PDF...")
    pdf_text = generator.read_pdf(args.pdf_path)

    print("Cleaning text...")
    cleaned_text = generator.clean_text(pdf_text, SYS_PROMPT_CLEANING)

    print("Generating podcast script...")
    podcast_script = generator.generate_script(cleaned_text, SYS_PROMPT_PODCAST)

    print("Rewriting podcast script...")
    rewritten_script = generator.generate_script(podcast_script, SYSTEM_PROMPT_REWRITE)

    print("Setting up Parler TTS model...")
    model, tokenizer = setup_parler_tts()

    print("Generating TTS audio with streaming...")
    chunk_size_in_s = 10  # Adjust as needed for optimal streaming performance
    generate_with_streaming(rewritten_script, "Will is speaking.", model, tokenizer, chunk_size_in_s, args.pdf_path)

if __name__ == "__main__":
    main()
