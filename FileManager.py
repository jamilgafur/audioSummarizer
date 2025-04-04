import os
import argparse
import torch
from transformers import AutoProcessor, BarkModel
import numpy as np
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import pdfplumber
from tqdm import tqdm
from transformers import pipeline
import pickle
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import warnings
import gc

warnings.filterwarnings('ignore')

# File Management Class
class FileManager:
    @staticmethod
    def read_pdf(file_path: str, max_chars: int = 100000):
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

    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")
        else:
            print("CUDA is not available. No memory cleared.")
