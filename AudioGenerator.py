from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

class AudioGenerator:
    def __init__(self):
        # Initialize the TTS pipeline with SpeechT5 model
        self.synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts")

        # Load speaker embedding (you can change the speaker by picking another index)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def generate_audio_chunk(self, text, chunk_index, output_path):
        """
        Generate audio for a given text chunk and save it to the specified output path.
        """
        # Use the synthesizer to generate speech for the given text chunk
        speech = self.synthesiser(text, forward_params={"speaker_embeddings": self.speaker_embedding})

        # Save the audio to the output path
        chunk_filename = f"{output_path}_part_{chunk_index}.wav"
        sf.write(chunk_filename, speech["audio"], samplerate=speech["sampling_rate"])
        print(f"Audio chunk saved to {chunk_filename}")

    def generate_audio_from_text(self, text, save_path="output_audio"):
        """
        Split the text into chunks (sentences or paragraphs), and generate separate audio files for each chunk.
        """
        # Split the text into sentences (you can modify this logic if you want to split by paragraphs or another criteria)
        sentences = text.split('. ')  # Basic sentence splitting

        # Generate audio for each sentence chunk
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                self.generate_audio_chunk(sentence, idx + 1, save_path)

        print(f"All audio chunks saved. Total: {len(sentences)} files.")
