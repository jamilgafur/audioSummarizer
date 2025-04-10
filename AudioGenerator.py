import os
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from kokoro import KPipeline
from nltk.tokenize import sent_tokenize
import os

nltk.download('punkt')

class AudioGenerator:
    def __init__(self):
        # Initialize Kokoro pipeline
        self.pipeline = KPipeline(lang_code='a')  # Set to English (adjust as needed)

    def generate_audio_from_text(self, text, save_path="output_audio", merged_output_file="final_podcast.wav", chunk_size=10):
        # Tokenize into sentences
        sentences = sent_tokenize(text)

        # Create the output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Remove first and last sentences for context trimming
        if len(sentences) > 2:
            sentences = sentences[1:-1]

        # Determine number of sentences per chunk
        total_sentences = len(sentences)
        sentences_per_chunk = max(1, total_sentences // chunk_size)

        chunk_list = [
            " ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, total_sentences, sentences_per_chunk)
        ]

        # Limit to exactly `chunk_size` chunks (in case of rounding mismatch)
        chunk_list = chunk_list[:chunk_size]

        # Generate audio chunks
        for idx, chunk in enumerate(chunk_list):
            if chunk.strip():
                self.generate_audio_chunk(chunk, idx + 1, save_path)

        # Merge the chunks into a single file
        self.merge_audio_chunks(save_path, merged_output_file)

    def generate_audio_chunk(self, text, index, save_path):
        """Generates a single audio file from a text chunk."""
        wav_data = self.pipeline.tts(text)
        out_path = os.path.join(save_path, f"chunk_{index}.wav")
        with open(out_path, "wb") as f:
            f.write(wav_data)
        print(f"Saved: {out_path}")

    def merge_audio_chunks(self, chunk_folder, output_file="final_podcast.wav"):
        """Merges all audio chunks in a folder into a single podcast file."""
        audio_files = sorted([f for f in os.listdir(chunk_folder) if f.endswith(".wav")])
        combined = AudioSegment.empty()
        for f in audio_files:
            audio = AudioSegment.from_wav(os.path.join(chunk_folder, f))
            combined += audio + AudioSegment.silent(duration=250)  # Add short pause
        combined.export(output_file, format="wav")
        print(f"Merged audio saved as: {output_file}")
