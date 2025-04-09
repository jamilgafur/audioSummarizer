from kokoro import KPipeline
import soundfile as sf
import os
import numpy as np
from pydub import AudioSegment
import re

class AudioGenerator:
    def __init__(self):
        # Initialize Kokoro pipeline
        self.pipeline = KPipeline(lang_code='a')  # Set to English (adjust as needed)

    def generate_audio_from_text(self, text, output_path='output.wav'):
        """
        Generate audio for a large text paragraph, split it into sentences,
        process each sentence, save each as a separate file, and merge at the end.
        """
        # Split the paragraph into sentences using a more robust method
        sentences = re.split(r'(?<!\w\.\w.)(?<=\.|\?)\s', text.strip())

        audio_files = []  # List to store the paths of generated audio files

        # Process each sentence, generate audio, and save to individual files
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                # Generate audio for the sentence using Kokoro
                generator = self.pipeline(sentence, voice='af_heart')  # You can choose other voices here
                for i, (gs, ps, audio) in enumerate(generator):
                    # Save the generated audio to a temporary file
                    temp_filename = f"temp_part_{idx + 1}.wav"
                    sf.write(temp_filename, audio, 24000)  # Save with a 24 kHz sample rate
                    print(f"Audio chunk for sentence {idx + 1} saved to {temp_filename}")
                    
                    # Add the temp filename to the list of audio files
                    audio_files.append(temp_filename)

        # Create an empty AudioSegment to hold all audio chunks
        final_audio = AudioSegment.empty()

        # Merge all the audio files into one final audio file using Pydub
        for audio_file in audio_files:
            # Load the audio file with Pydub
            audio_chunk = AudioSegment.from_wav(audio_file)

            # Append the audio chunk to the final_audio
            final_audio += audio_chunk

            # Optionally, delete the temporary audio file after reading it
            os.remove(audio_file)

        # Export the merged audio to the final output path
        final_audio.export(output_path, format="wav")
        print(f"All audio chunks merged and saved to {output_path}")

