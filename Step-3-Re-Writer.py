import argparse
import torch
import pickle
import warnings
from transformers import pipeline

warnings.filterwarnings('ignore')

SYSTEM_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

REMEMBER THIS WITH YOUR HEART
The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""

def load_input(input_path: str):
    with open(input_path, 'rb') as file:
        return pickle.load(file)

def save_output(output_path: str, data):
    with open(output_path, 'wb') as file:
        pickle.dump(data, file)

def generate_podcast(
    model_name: str,
    input_prompt,
    max_new_tokens: int = 8126,
    temperature: float = 1.0
):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_prompt},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return outputs[0]["generated_text"][-1]['content']

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
    generated_text = generate_podcast(
        model_name=args.model,
        input_prompt=input_prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )

    print(f"Saving output to: {args.output}")
    save_output(args.output, generated_text)
    print("Done!")

if __name__ == "__main__":
    main()