#!/usr/bin/env python
# coding: utf-8

# ## Notebook 4: TTS Workflow
# 
# We have the exact podcast transcripts ready now to generate our audio for the Podcast.
# 
# In this notebook, we will learn how to generate Audio using both `suno/bark` and `parler-tts/parler-tts-mini-v1` models first. 
# 
# After that, we will use the output from Notebook 3 to generate our complete podcast
# 
# Note: Please feel free to extend this notebook with newer models. The above two were chosen after some tests using a sample prompt.

# ⚠️ Warning: This notebook likes have `transformers` version to be `4.43.3` or earlier so we will downgrade our environment to make sure things run smoothly

# Credit: [This](https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing#scrollTo=68QtoUqPWdLk) Colab was used for starter code
# 

# We can install these packages for speedups

# In[1]:


#!pip3 install optimum
#!pip install -U flash-attn --no-build-isolation
#!pip install transformers==4.43.3


# Let's import the necessary frameworks

# In[2]:


from IPython.display import Audio
import IPython.display as ipd
from tqdm import tqdm


# In[3]:


from transformers import BarkModel, AutoProcessor, AutoTokenizer
import torch
import json
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration


# ### Testing the Audio Generation

# Let's try generating audio using both the models to understand how they work. 
# 
# Note the subtle differences in prompting:
# - Parler: Takes in a `description` prompt that can be used to set the speaker profile and generation speeds
# - Suno: Takes in expression words like `[sigh]`, `[laughs]` etc. You can find more notes on the experiments that were run for this notebook in the [TTS_Notes.md](./TTS_Notes.md) file to learn more.

# Please set `device = "cuda"` below if you're using a single GPU node.

# #### Parler Model
# 
# Let's try using the Parler Model first and generate a short segment with speaker Laura's voice

# In[7]:


# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Define text and description
text_prompt = """
Exactly! And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
"""
description = """
Laura's voice is expressive and dramatic in delivery, speaking at a fast pace with a very close recording that almost has no background noise.
"""
# Tokenize inputs
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)

# Generate audio
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

# Play audio in notebook
ipd.Audio(audio_arr, rate=model.config.sampling_rate)


# #### Bark Model
# 
# Amazing, let's try the same with bark now:
# - We will set the `voice_preset` to our favorite speaker
# - This time we can include expression prompts inside our generation prompt
# - Note you can CAPTILISE words to make the model emphasise on these
# - You can add hyphens to make the model pause on certain words

# In[9]:


voice_preset = "v2/en_speaker_6"
sampling_rate = 24000


# In[8]:


device = "cuda:7"

processor = AutoProcessor.from_pretrained("suno/bark")

#model =  model.to_bettertransformer()
#model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)#.to_bettertransformer()


# In[11]:


text_prompt = """
Exactly! [sigh] And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
"""
inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)


# ## Bringing it together: Making the Podcast
# 
# Okay now that we understand everything-we can now use the complete pipeline to generate the entire podcast
# 
# Let's load in our pickle file from earlier and proceed:

# In[3]:


import pickle

with open('./resources/podcast_ready_data.pkl', 'rb') as file:
    PODCAST_TEXT = pickle.load(file)


# Let's define load in the bark model and set it's hyper-parameters for discussions

# In[4]:


bark_processor = AutoProcessor.from_pretrained("suno/bark")
bark_model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to("cuda:3")
bark_sampling_rate = 24000


# Now for the Parler model:

# In[5]:


parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to("cuda:3")
parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")


# In[6]:


speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""


# We will concatenate the generated segments of audio and also their respective sampling rates since we will require this to generate the final audio

# In[7]:


generated_segments = []
sampling_rates = []  # We'll need to keep track of sampling rates for each segment


# In[8]:


device="cuda:3"


# Function generate text for speaker 1

# In[9]:


def generate_speaker1_audio(text):
    """Generate audio using ParlerTTS for Speaker 1"""
    input_ids = parler_tokenizer(speaker1_description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, parler_model.config.sampling_rate


# Function to generate text for speaker 2

# In[10]:


def generate_speaker2_audio(text):
    """Generate audio using Bark for Speaker 2"""
    inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
    speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    audio_arr = speech_output[0].cpu().numpy()
    return audio_arr, bark_sampling_rate


# Helper function to convert the numpy output from the models into audio

# In[38]:


def numpy_to_audio_segment(audio_arr, sampling_rate):
    """Convert numpy array to AudioSegment"""
    # Convert to 16-bit PCM
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    
    # Create WAV file in memory
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    
    # Convert to AudioSegment
    return AudioSegment.from_wav(byte_io)


# In[16]:


PODCAST_TEXT


# Most of the times we argue in life that Data Structures isn't very useful. However, this time the knowledge comes in handy. 
# 
# We will take the string from the pickle file and load it in as a Tuple with the help of `ast.literal_eval()`

# In[18]:


import ast
ast.literal_eval(PODCAST_TEXT)


# #### Generating the Final Podcast
# 
# Finally, we can loop over the Tuple and use our helper functions to generate the audio

# In[39]:


final_audio = None

for speaker, text in tqdm(ast.literal_eval(PODCAST_TEXT), desc="Generating podcast segments", unit="segment"):
    if speaker == "Speaker 1":
        audio_arr, rate = generate_speaker1_audio(text)
    else:  # Speaker 2
        audio_arr, rate = generate_speaker2_audio(text)
    
    # Convert to AudioSegment (pydub will handle sample rate conversion automatically)
    audio_segment = numpy_to_audio_segment(audio_arr, rate)
    
    # Add to final audio
    if final_audio is None:
        final_audio = audio_segment
    else:
        final_audio += audio_segment


# ### Output the Podcast
# 
# We can now save this as a mp3 file

# In[40]:


final_audio.export("./resources/_podcast.mp3", 
                  format="mp3", 
                  bitrate="192k",
                  parameters=["-q:a", "0"])


# ### Suggested Next Steps:
# 
# - Experiment with the prompts: Please feel free to experiment with the SYSTEM_PROMPT in the notebooks
# - Extend workflow beyond two speakers
# - Test other TTS Models
# - Experiment with Speech Enhancer models as a step 5.

# In[ ]:


#fin

