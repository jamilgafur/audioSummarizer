#!/usr/bin/env python
# coding: utf-8

# ## Notebook 3: Transcript Re-writer
# 
# In the previous notebook, we got a great podcast transcript using the raw file we have uploaded earlier. 
# 
# In this one, we will use `Llama-3.1-8B-Instruct` model to re-write the output from previous pipeline and make it more dramatic or realistic.

# We will again set the `SYSTEM_PROMPT` and remind the model of its task. 
# 
# Note: We can even prompt the model like so to encourage creativity:
# 
# > Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.
# 

# Note: We will prompt the model to return a list of Tuples to make our life easy in the next stage of using these for Text To Speech Generation

# In[1]:


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


# This time we will use the smaller 8B model

# In[2]:


MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# Let's import the necessary libraries

# In[3]:


# Import necessary libraries
import torch
from accelerate import Accelerator
import transformers

from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')


# We will load in the pickle file saved from previous notebook
# 
# This time the `INPUT_PROMPT` to the model will be the output from the previous stage

# In[4]:


import pickle

with open('./resources/data.pkl', 'rb') as file:
    INPUT_PROMPT = pickle.load(file)


# We can again use Hugging Face `pipeline` method to generate text from the model

# In[ ]:


pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": INPUT_PROMPT},
]

outputs = pipeline(
    messages,
    max_new_tokens=8126,
    temperature=1,
)


# We can verify the output from the model

# In[ ]:


print(outputs[0]["generated_text"][-1])


# In[ ]:


save_string_pkl = outputs[0]["generated_text"][-1]['content']


# Let's save the output as a pickle file to be used in Notebook 4

# In[ ]:


with open('./resources/podcast_ready_data.pkl', 'wb') as file:
    pickle.dump(save_string_pkl, file)


# ### Next Notebook: TTS Workflow
# 
# Now that we have our transcript ready, we are ready to generate the audio in the next notebook.

# In[ ]:


#fin

