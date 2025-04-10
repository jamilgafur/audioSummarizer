import os
import nltk
import soundfile as sf
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from kokoro import KPipeline

nltk.download('punkt')

class AudioGenerator:
    def __init__(self, voice='af_heart'):
        self.pipeline = KPipeline(lang_code='a')
        self.voice = voice

    def generate_audio_from_text(
        self,
        text,
        output_file="final_output.wav",
        save_path="output_audio",
        silence_duration_ms=300
    ):
        # Tokenize the entire text into sentences
        sentences = sent_tokenize(text)
        os.makedirs(save_path, exist_ok=True)

        final_audio = AudioSegment.empty()

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 3:
                print(f"Skipping sentence {i+1}: too short or empty.")
                continue

            try:
                generator = self.pipeline(sentence, voice=self.voice)
                for _, _, audio in generator:
                    temp_path = os.path.join(save_path, "temp.wav")
                    sf.write(temp_path, audio.numpy(), 24000)
                    segment = AudioSegment.from_wav(temp_path)
                    final_audio += segment + AudioSegment.silent(duration=silence_duration_ms)
                    print(f"Processed sentence {i+1}")
                    break
            except Exception as e:
                print(f"Error on sentence {i+1}: {e}")

        if len(final_audio) > 0:
            final_path = os.path.join(save_path, output_file)
            final_audio.export(final_path, format="wav")
            print(f"Final audio saved as: {final_path}")
        else:
            print("No audio was generated.")

