# Podcast TTS Workflow

This repository implements an end-to-end pipeline for converting PDF documents into a podcast-ready audio file. The solution strictly follows the provided design: each processing stage is handled by a dedicated Python script, using Hugging Face Transformers for text generation and Parler TTS for audio synthesis. This workflow is designed to work with specific input formats and configuration parameters as detailed below.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Detailed Workflow](#detailed-workflow)
  - [Step 1: PDF Pre-Processing](#step-1-pdf-pre-processing)
  - [Step 2: Transcript Writer](#step-2-transcript-writer)
  - [Step 3: Re-Writer (Single Speaker Monologue)](#step-3-re-writer-single-speaker-monologue)
  - [Step 4: TTS Workflow](#step-4-tts-workflow)
- [Code Citation](#code-citation)
- [License](#license)

---

## Overview

The pipeline converts PDF content into an audio podcast through four major stages:

1. **PDF Pre-Processing:**  
   - Validates the input PDF.
   - Extracts text using `pdfplumber` and cleans encoding issues.
   - Splits the extracted text into manageable chunks (controlled by a maximum character limit).
   - Processes each chunk with a Transformer model to produce cleaned and formatted text.

2. **Transcript Writer:**  
   - Reads the cleaned text output from Step 1.
   - Uses a text-generation pipeline with a detailed system prompt to generate a podcast transcript.
   - The transcript is designed to be engaging and mimics a realistic, flowing conversation as per the provided instructions.

3. **Re-Writer (Single Speaker Monologue):**  
   - Transforms the initial transcript into a cohesive, single-speaker monologue.
   - The system prompt instructs the model to remove any dialogue markers or multiple speaker labels, ensuring a continuous narrative.
   - This transformation is critical for generating a final script that is compatible with AI Text-To-Speech (TTS) pipelines.

4. **TTS Workflow:**  
   - Converts the monologue transcript into audio using Parler TTS.
   - Generates individual audio segments for each part of the transcript.
   - Combines these segments into the final podcast audio output.
   - The process strictly follows the provided instructions without adding extra formatting or markers.

---

## Directory Structure

The repository is organized into four main Python scripts, each corresponding to a step in the workflow:

```plaintext
└── ./
    ├── Step-1 PDF-Pre-Processing-Logic.py
    ├── Step-2-Transcript-Writer.py
    ├── Step-3-Re-Writer.py
    └── Step-4-TTS-Workflow.py