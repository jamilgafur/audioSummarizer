import argparse
import torch
import pickle
from transformers import pipeline, AutoProcessor, AutoTokenizer


import warnings
import io
import numpy as np
from tqdm import tqdm
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from pydub import AudioSegment
from scipy.io import wavfile
import os

warnings.filterwarnings('ignore')

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define system prompt for generating the podcast
SYSTEM_PROMPT = """
You are an experienced and engaging podcast host. Your job is to deliver insightful, informative, and captivating content to listeners. 

The podcast will feature you, the sole speaker. Your tone should be confident, yet approachable, using clear and concise explanations, while also offering interesting anecdotes and analogies to make the content accessible and engaging.

You are talking directly to the audience, guiding them through the concepts, and making sure to keep the conversation lively. The style is informative, educational, and exciting. Ensure that the content flows smoothly and keeps the listener intrigued. Add enthusiasm to your delivery to make the technical details more engaging.

START YOUR RESPONSE DIRECTLY WITH THE SCRIPT:

Welcome Everybody...
"""

# Function to load the input pickle file
def load_input(input_path: str):
    with open(input_path, 'rb') as file:
        return pickle.load(file)

# Function to save the output to a pickle file
def save_output(output_path: str, data):
    with open(output_path, 'wb') as file:
        pickle.dump(data, file)

# Function to generate podcast content
def generate_podcast(model_name: str, input_prompt, max_new_tokens: int = 8126, temperature: float = 1.0):
    pipe = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": input_prompt}]
    outputs = pipe(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return outputs[0]["generated_text"][-1]['content']

# Setup Parler TTS models
def setup_parler_tts():
    # Load Parler TTS model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    return model, tokenizer

# Generate audio for Jon's voice (Parler TTS)
def generate_audio_parler(model, tokenizer, text, description="Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, model.config.sampling_rate

# Convert numpy array to AudioSegment
def numpy_to_audio_segment(audio_arr, sampling_rate):
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    return AudioSegment.from_wav(byte_io)

# Save each audio segment to a file for validation
def save_audio_segment(audio_segment, index, output_dir):
    filename = os.path.join(output_dir, f"segment_{index}.wav")
    audio_segment.export(filename, format="wav")

# Main function for podcast generation
def generate_podcast_audio(podcast_data, output_dir="generated_audio"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_audio = None
    model, tokenizer = setup_parler_tts()

    for index, text in tqdm(enumerate(podcast_data), desc="Generating podcast segments", unit="segment"):
        # Generate audio for each segment with Jon's voice
        audio_arr, rate = generate_audio_parler(model, tokenizer, text)

        # Convert audio to AudioSegment
        audio_segment = numpy_to_audio_segment(audio_arr, rate)
        
        # Save each segment for validation
        save_audio_segment(audio_segment, index, output_dir)

        # Combine the audio segments
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment

    return final_audio

# Save final audio as an MP3
def save_podcast_audio(final_audio, output_path="./resources/_single_speaker_podcast.mp3"):
    final_audio.export(output_path, format="mp3", bitrate="192k", parameters=["-q:a", "0"])

# Main execution logic
def main():
    parser = argparse.ArgumentParser(description="Reformat podcast script using HuggingFace models")
    parser.add_argument('--input', type=str, default='./resources/data.pkl', help='Path to the input pickle file')
    parser.add_argument('--output', type=str, default='./resources/podcast_ready_data.pkl', help='Path to save the podcast-ready output')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', help='HuggingFace model to use')
    parser.add_argument('--max_tokens', type=int, default=8126, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')

    args = parser.parse_args()

    print(f"Loading input from: {args.input}")
    input_prompt = load_input(args.input)

    print("Generating podcast content...")
    generated_text = generate_podcast(model_name=args.model, input_prompt=input_prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)

    print("Generating final podcast...")
    podcast_data = generated_text.split('\n')  # Assuming each line is a podcast segment
    final_audio = generate_podcast_audio(podcast_data)

    print(f"Saving output to: {args.output}")
    save_podcast_audio(final_audio)
    save_output(args.output, podcast_data)

    print("Done!")

if __name__ == "__main__":
    main()
