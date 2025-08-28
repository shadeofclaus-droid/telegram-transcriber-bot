
# Telegram Transcription Bot

This project contains a Telegram bot that transcribes voice notes, audio, video, or audio files using OpenAI's Whisper API. It includes features such as logging, language selection, file size limits, privacy statements, and additional commands like `/ping` and `/privacy`.

## Setup

### Local Setup

1. Install Python 3.11 or later.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`.
4. Run the bot:
   ```bash
   python bot.py
   ```

### Docker Deployment

A `Dockerfile` is provided to build an image that includes FFmpeg and necessary dependencies. To build and run the container locally:

```bash
docker build -t tg-transcriber .
docker run -e TELEGRAM_BOT_TOKEN=your_token -e OPENAI_API_KEY=your_key tg-transcriber
```

### Deployment to Railway

1. Push this project to a Git repository.
2. In Railway, create a new project and connect it to your repository.
3. Set environment variables `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY` in the Variables section.
4. Deploy the service; it will run the bot using polling.

## Usage

Send voice notes, audio files, or videos to the bot in Telegram. The bot will transcribe the audio to text. You can specify the language by adding `lang=uk` or `lang=en` in the caption of your message. If the transcription is lengthy, the bot sends it as a text file.

### Commands

- `/start` – Greeting and basic instructions.
- `/help` – Shows usage instructions.
- `/ping` – Simple health check (bot responds with “pong”).
- `/privacy` – Short privacy policy describing how files are handled.

## Notes

- Ensure FFmpeg is installed in your environment; the provided Dockerfile handles this automatically.
- The file size limit is set to 200 MB; adjust `MAX_FILE_BYTES` in `bot.py` if needed.
- The bot uses OpenAI's Whisper API; ensure your API key has sufficient quota.
