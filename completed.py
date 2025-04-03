import os
import argparse
import torch
import pdfplumber
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle
import warnings
from parler_tts import ParlerTTSForConditionalGeneration
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

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

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
        # Define the conversation with system and user messages
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ]
        
        # Extract the text from the conversation to feed to the tokenizer
        conversation_text = "\n".join([msg["content"] for msg in conversation])
        
        # Tokenize the conversation text
        inputs = self.pipeline.tokenizer(conversation_text, return_tensors="pt").to(self.pipeline.device)
        
        # Generate the output using the model
        with torch.no_grad():
            output = self.pipeline.model.generate(**inputs, temperature=self.temperature, top_p=0.9, max_new_tokens=512)
        
        # Decode the output to text and return it
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
    # Assume this sets up the TTS model and tokenizer, implement as needed.
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    return model, tokenizer


def numpy_to_audio_segment(audio_data, rate):
    # Check if CUDA is available, else fallback to CPU
    device = "cpu"

    # If audio_data is a PyTorch tensor, move it to the correct device
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.to(device)
    
    # Convert the tensor to a numpy array and normalize the values
    audio_arr = audio_data.cpu().numpy() if isinstance(audio_data, torch.Tensor) else np.array(audio_data, dtype=np.float32)
    audio_arr = np.interp(audio_arr, (audio_arr.min(), audio_arr.max()), (-1, 1))  # Normalize to -1 to 1

    # Convert the numpy array into an AudioSegment
    audio_segment = AudioSegment(
        audio_arr.tobytes(), 
        frame_rate=rate, 
        sample_width=audio_arr.itemsize, 
        channels=1
    )
    return audio_segment

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

    print("Generating TTS audio...")
    model, tokenizer = setup_parler_tts()
    audio_data, rate = generator.generate_audio(rewritten_script, model, tokenizer)

    print("Converting to audio segment...")
    audio_segment = numpy_to_audio_segment(audio_data, rate)

    print("Podcast generation complete!")


if __name__ == "__main__":
    main()
