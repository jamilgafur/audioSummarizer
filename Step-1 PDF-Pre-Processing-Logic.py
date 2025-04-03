#!/usr/bin/env python
# coding: utf-8

# ## Notebook 1: PDF Pre-processing

# In the series, we will be going from a PDF to Podcast using all open models. 
# 
# The first step in getting to the podcast is finding a script, right now our logic is:
# - Use any PDF on any topic
# - Prompt `Llama-3.2-1B-Instruct` model to process it into a text file
# - Re-write this into a podcast transcript in next notebook.
# 
# In this notebook, we will upload a PDF and save it into a `.txt` file using the `PyPDF2` library, later we will process chunks from the text file using our featherlight model.

# Most of us shift-enter pass the comments to realise later we need to install libraries. For the few that read the instructions, please remember to do so:

# In[41]:


#!pip install PyPDF2
#!pip install rich ipywidgets


# Assuming you have a PDF uploaded on the same machine, please set the path for the file. 
# 
# Also, if you want to flex your GPU-please switch to a bigger model although the featherlight models work perfectly for this task:

# In[14]:


pdf_path = './resources/2402.13116v4.pdf'
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# In[49]:


import PyPDF2
from typing import Optional
import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')


# Let's make sure we don't stub our toe by checking if the file exists

# In[9]:


def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print("Error: File is not a PDF")
        return False
    return True


# Convert PDF to a `.txt` file. This would simply read and dump the contents of the file. We set the maximum characters to 100k. 
# 
# For people converting their favorite novels into a podcast, they will have to add extra logic of going outside the Llama models context length which is 128k tokens.

# In[10]:


def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> Optional[str]:
    if not validate_pdf(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")
            
            extracted_text = []
            total_chars = 0
            
            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break
                
                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")
            
            final_text = '\n'.join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text
            
    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


# Helper function to grab meta info about our PDF

# In[11]:


# Get PDF metadata
def get_pdf_metadata(file_path: str) -> Optional[dict]:
    if not validate_pdf(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata
            }
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None


# Finally, we can run our logic to extract the details from the file

# In[12]:


# Extract metadata first
print("Extracting metadata...")
metadata = get_pdf_metadata(pdf_path)
if metadata:
    print("\nPDF Metadata:")
    print(f"Number of pages: {metadata['num_pages']}")
    print("Document info:")
    for key, value in metadata['metadata'].items():
        print(f"{key}: {value}")

# Extract text
print("\nExtracting text...")
extracted_text = extract_text_from_pdf(pdf_path)

# Display first 500 characters of extracted text as preview
if extracted_text:
    print("\nPreview of extracted text (first 500 characters):")
    print("-" * 50)
    print(extracted_text[:500])
    print("-" * 50)
    print(f"\nTotal characters extracted: {len(extracted_text)}")

# Optional: Save the extracted text to a file
if extracted_text:
    output_file = 'extracted_text.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"\nExtracted text has been saved to {output_file}")


# ### Llama Pre-Processing
# 
# Now let's proceed to justify our distaste for writing regex and use that as a justification for a LLM instead:
# 
# At this point, have a text file extracted from a PDF of a paper. Generally PDF extracts can be messy due to characters, formatting, Latex, Tables, etc. 
# 
# One way to handle this would be using regex, instead we can also prompt the feather light Llama models to clean up our text for us. 
# 
# Please try changing the `SYS_PROMPT` below to see what improvements you can make:

# In[60]:


device = "cuda" if torch.cuda.is_available() else "cpu"

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


# Instead of having the model process the entire file at once, as you noticed in the prompt-we will pass chunks of the file. 
# 
# One issue with passing chunks counted by characters is, we lose meaning of words so instead we chunk by words:

# In[61]:


def create_word_bounded_chunks(text, target_chunk_size):
    """
    Split text into chunks at word boundaries close to the target chunk size.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
            # Join the current chunk and add it to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# Let's load in the model and start processing the text chunks

# In[62]:


accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
model, tokenizer = accelerator.prepare(model, tokenizer)


# In[63]:


def process_chunk(text_chunk, chunk_num):
    """Process a chunk of text and return both input and output for verification"""
    conversation = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text_chunk},
    ]
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512
        )
    
    processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
    
    # Print chunk information for monitoring
    #print(f"\n{'='*40} Chunk {chunk_num} {'='*40}")
    print(f"INPUT TEXT:\n{text_chunk[:500]}...")  # Show first 500 chars of input
    print(f"\nPROCESSED TEXT:\n{processed_text[:500]}...")  # Show first 500 chars of output
    print(f"{'='*90}\n")
    
    return processed_text


# In[64]:


INPUT_FILE = "./resources/extracted_text.txt"  # Replace with your file path
CHUNK_SIZE = 1000  # Adjust chunk size if needed

chunks = create_word_bounded_chunks(text, CHUNK_SIZE)
num_chunks = len(chunks)


# In[65]:


num_chunks


# In[66]:


# Read the file
with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    text = file.read()

# Calculate number of chunks
num_chunks = (len(text) + CHUNK_SIZE - 1) // CHUNK_SIZE

# Cell 6: Process the file with ordered output
# Create output file name
output_file = f"clean_{os.path.basename(INPUT_FILE)}"


# In[67]:


with open(output_file, 'w', encoding='utf-8') as out_file:
    for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Process chunk and append to complete text
        processed_chunk = process_chunk(chunk, chunk_num)
        processed_text += processed_chunk + "\n"
        
        # Write chunk immediately to file
        out_file.write(processed_chunk + "\n")
        out_file.flush()


# Let's print out the final processed versions to make sure things look good

# In[68]:


print(f"\nProcessing complete!")
print(f"Input file: {INPUT_FILE}")
print(f"Output file: {output_file}")
print(f"Total chunks processed: {num_chunks}")

# Preview the beginning and end of the complete processed text
print("\nPreview of final processed text:")
print("\nBEGINNING:")
print(processed_text[:1000])
print("\n...\n\nEND:")
print(processed_text[-1000:])


# ### Next Notebook: Transcript Writer
# 
# Now that we have the pre-processed text ready, we can move to converting into a transcript in the next notebook

# In[ ]:


#fin

