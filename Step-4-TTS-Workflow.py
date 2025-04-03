from IPython.display import Audio
import IPython.display as ipd
from tqdm import tqdm

from transformers import BarkModel, AutoProcessor, AutoTokenizer
import torch
import json
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration

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

voice_preset = "v2/en_speaker_6"
sampling_rate = 24000

device = "cuda:7"

processor = AutoProcessor.from_pretrained("suno/bark")

model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)#.to_bettertransformer()

text_prompt = """
Exactly! [sigh] And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
"""
inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)

import pickle

with open('./resources/podcast_ready_data.pkl', 'rb') as file:
    PODCAST_TEXT = pickle.load(file)

bark_processor = AutoProcessor.from_pretrained("suno/bark")
bark_model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to("cuda:3")
bark_sampling_rate = 24000

parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to("cuda:3")
parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""

generated_segments = []
sampling_rates = []  # We'll need to keep track of sampling rates for each segment

device="cuda:3"

def generate_speaker1_audio(text):
    """Generate audio using ParlerTTS for Speaker 1"""
    input_ids = parler_tokenizer(speaker1_description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, parler_model.config.sampling_rate

def generate_speaker2_audio(text):
    """Generate audio using Bark for Speaker 2"""
    inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
    speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    audio_arr = speech_output[0].cpu().numpy()
    return audio_arr, bark_sampling_rate

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

