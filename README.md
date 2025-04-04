Here is the README for the `audioSummarizer` repository based on the provided code structure:

---

# Audio Summarizer

The **Audio Summarizer** package converts PDF text into a podcast-ready audio format. This package extracts text from PDF files, processes the content using language models, and generates high-quality, engaging audio using Text-to-Speech (TTS) models.

## Features

- **PDF Text Extraction**: Extracts text from PDF files using the `pdfplumber` library.
- **Text Cleaning and Processing**: Cleans and processes extracted text to make it more suitable for podcast narration.
- **Podcast Script Generation**: Uses Hugging Face and Ollama models to rewrite the extracted text into a conversational podcast script.
- **Audio Generation**: Converts the podcast script into audio using the SpeechT5 TTS model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

To use the **Audio Summarizer** package, ensure you have Python 3.7+ and pip installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To generate an audio podcast from a PDF document, run the following command:

```bash
python completed.py <pdf_path> [options]
```

### Arguments

- `pdf_path`: The path to the PDF file you want to convert to audio.
- `--modelOllama`: Specify the Ollama model to use for text processing (default: `llama3.2`).
- `--modelHugging`: Specify the Hugging Face model for text generation (default: `facebook/opt-125m`).
- `--max_tokens`: Set the maximum number of tokens for the generation (default: `8126`).
- `--temperature`: Set the sampling temperature for text generation (default: `1.0`).
- `--pdf_text_file`: The filename to save extracted PDF text (default: `pdf_text.txt`).
- `--cleaned_text_file`: The filename to save cleaned text (default: `cleaned.txt`).
- `--audiomodelname`: The TTS model to use for generating the audio (default: `suno/bark-small`).

### Example

```bash
python completed.py my_document.pdf --modelOllama llama3.2 --modelHugging facebook/opt-125m --audiomodelname suno/bark-small
```

This will process the PDF `my_document.pdf`, clean and rewrite the text, and generate audio output in the `output_audio` directory.

## Directory Structure

```plaintext
./
├── AudioGenerator.py      # Handles audio generation using TTS models
├── completed.py           # Main script for PDF processing and audio generation
├── FileManager.py         # Manages file reading, cleaning, and GPU memory management
└── QueryHandler.py        # Handles text processing, chunking, and interaction with Hugging Face and Ollama models
```

### Detailed Workflow

1. **PDF Pre-Processing**: Extract text from the PDF using the `FileManager.read_pdf` method.
2. **Text Cleaning and Script Generation**: The `TextProcessor.clean_text` method cleans the extracted text and generates a conversational podcast script using Hugging Face and Ollama models.
3. **Audio Generation**: The `AudioGenerator.generate_audio_from_text` method splits the text into chunks and generates audio using the SpeechT5 TTS model.

## Dependencies

- `torch`: Required for deep learning models and TTS pipeline.
- `transformers`: Hugging Face library for using pre-trained models.
- `datasets`: Hugging Face dataset library for loading speaker embeddings.
- `soundfile`: For saving the generated audio to files.
- `pydub`: Used for audio manipulation (if necessary).
- `pdfplumber`: Used to extract text from PDF files.

### Install Dependencies

You can install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to customize the README further if needed!