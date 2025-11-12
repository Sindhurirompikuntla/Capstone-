# Audio Sample Data

## Note on Audio Files

This directory is intended to contain sample audio files of sales conversations. Due to file size constraints and the nature of this setup, actual audio files are not included in this repository.

## Creating Sample Audio Files

To create sample audio files for testing, you can:

1. **Use Text-to-Speech Services:**
   - Azure Text-to-Speech API
   - Google Cloud Text-to-Speech
   - Amazon Polly
   - Online TTS tools

2. **Record Your Own Samples:**
   - Use the text transcripts in `data/text/` as scripts
   - Record using any audio recording software
   - Save in supported formats: MP3, WAV, M4A, or OGG

3. **Use the Provided Script:**
   - A Python script is available in `examples/generate_audio_samples.py`
   - This script can convert text transcripts to audio using Azure TTS

## Supported Audio Formats

- MP3
- WAV
- M4A
- OGG

## Maximum File Size

- 25 MB per file (configurable in `config/config.yaml`)

## Sample File Naming Convention

- `sample_conversation_1.mp3`
- `sample_conversation_2.wav`
- `sales_call_crm_demo.mp3`
- `security_solution_pitch.wav`

## Testing Without Audio Files

You can test the system using the text transcripts in `data/text/` directory:
- `sample_transcript_1.txt` - CRM Solution Sales Call
- `sample_transcript_2.txt` - Cybersecurity Solution Sales Call
- `sample_transcript_3.txt` - Marketing Analytics Sales Call

