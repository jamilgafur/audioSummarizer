import argparse
import torch
import pickle
import warnings
from transformers import pipeline
from pathlib import Path

warnings.filterwarnings('ignore')

SYSTEM_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
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

    def read_file_to_string(self, filename):
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode file '{filename}' using common encodings.")

    def generate_script(self, input_text):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return outputs[0]["generated_text"][-1]['content']

    def save_to_pickle(self, output_data, output_file):
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

def main():
    parser = argparse.ArgumentParser(description="Generate podcast script from input text using Hugging Face LLMs.")
    parser.add_argument('--input', default='./resources/clean_extracted_text.txt', help='Path to the input .txt file')
    parser.add_argument('--output', default='./resources/data.pkl', help='Path to save the output .pkl file')

    parser.add_argument('--model', default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help='Model to use from Hugging Face')
    parser.add_argument('--max_tokens', type=int, default=8126, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for generation')

    args = parser.parse_args()

    generator = PodcastGenerator(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    try:
        input_text = generator.read_file_to_string(args.input)
        script = generator.generate_script(input_text)
        print(script)
        generator.save_to_pickle(script, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()