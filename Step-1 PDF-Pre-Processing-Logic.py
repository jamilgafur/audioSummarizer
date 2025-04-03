import os
import argparse
import pdfplumber
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def validate_pdf(file_path: str) -> bool:
    """Validate if the given file path points to a PDF file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print("Error: File is not a PDF")
        return False
    return True

def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> str:
    """Extract text from a PDF file using pdfplumber with encoding fixes."""
    if not validate_pdf(file_path):
        return ""

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

def create_chunks(text, chunk_size):
    """Split text into chunks of a specified size."""
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_text_chunk(model, tokenizer, text_chunk, sys_prompt, device):
    """Process a text chunk using a Transformer model."""
    conversation = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text_chunk},
    ]
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=512)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

def main():
    parser = argparse.ArgumentParser(description="Process a PDF into cleaned text using Transformers")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--output", type=str, default="cleaned_text.txt", help="Output text file name")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Hugging Face model to use")
    parser.add_argument("--chunk_size", type=int, default=20000, help="Size of text chunks for processing")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True  # Ensures remote code execution without prompt
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Extracting text from {args.pdf_path}...")
    text = extract_text_from_pdf(args.pdf_path)
    if not text:
        print("Failed to extract text.")
        return
    
    chunks = create_chunks(text, args.chunk_size)

    SYS_PROMPT = """
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

    print(f"Processing {len(chunks)} chunks...")
    processed_text = ""

    with open(args.output, "w", encoding="utf-8") as out_file:
        for chunk in tqdm(chunks, desc="Processing Chunks"):
            cleaned_chunk = process_text_chunk(model, tokenizer, chunk, SYS_PROMPT, device)
            out_file.write(cleaned_chunk + "\n")
            processed_text += cleaned_chunk + "\n"
            print(chunk)
            print("----"*10)
            print(cleaned_chunk)
            print("---"*10)
    
    print(f"Processing complete! Cleaned text saved to {args.output}")

if __name__ == "__main__":
    main()
