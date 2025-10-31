# AI Wholesaler Bot

A voice AI bot for "We Buy Houses" lead generation. The bot conducts conversations with potential sellers, collects property details, and automatically sends lead summaries via email when calls end.

## Features

- Voice conversation with potential property sellers
- Gathers property information (address, condition, reason for selling, timeline, asking price)
- Books follow-up calls with specialists
- Automatically sends lead summary emails when participants disconnect

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

### Required API Keys

You'll need API keys from the following services:

- [Daily](https://www.daily.co/) - For WebRTC rooms and video/audio
- [OpenAI](https://auth.openai.com/create-account) - For LLM inference
- [ElevenLabs](https://elevenlabs.io/) - For Text-to-Speech (API key + Voice ID)
- [Simli](https://simli.ai/) - For AI avatar (API key + Face ID)

### Email Configuration

For lead notifications, configure SMTP settings:

- SMTP_USER - Email address for sending
- SMTP_PASS - SMTP password
- SMTP_HOST - SMTP server (e.g., smtp.gmail.com)
- SMTP_PORT - SMTP port (e.g., 465)
- LEAD_NOTIFY_EMAIL - Email address to receive lead notifications
- DAILY_API_URL - (Optional) Defaults to https://api.daily.co/v1

## Setup

1. Clone this repository:

   ```bash
   git clone <your-repo-url>
   cd pipecat-quickstart
   ```

2. Create a `.env` file with your API keys:

   ```ini
   DAILY_API_KEY=your_daily_api_key
   OPENAI_API_KEY=your_openai_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id
   SIMLI_API_KEY=your_simli_api_key
   SIMLI_FACE_ID=your_simli_face_id
   SMTP_USER=your_email@example.com
   SMTP_PASS=your_smtp_password
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=465
   LEAD_NOTIFY_EMAIL=recipient@example.com
   ```

3. Install dependencies:

   ```bash
   uv sync
   ```

## Running the Bot

Start the bot runner:

```bash
uv run backend/bot_runner.py --host 127.0.0.1 --port 7860
```

Or with auto-reload for development:

```bash
uv run backend/bot_runner.py --host 127.0.0.1 --port 7860 --reload
```

**Open http://127.0.0.1:7860 in your browser** and click `Start Session` to provision a Daily room and launch the worker. You'll get a room URL and token to join.

> 💡 First run note: The initial startup may take ~20 seconds as Pipecat downloads required models and imports.

## How It Works

1. User clicks "Start Session" which creates a Daily room and starts the bot worker
2. User joins the room and talks with the AI agent "Alex"
3. Alex collects property information (condition, reason for selling, timeline, asking price)
4. When the participant leaves, the bot automatically:
   - Extracts the conversation transcript
   - Uses AI to write a summary email to Jimmy (the specialist)
   - Sends the email with lead details

## Customization

- **Bot personality**: Edit the system message in `backend/bot.py` (line 75)
- **Email format**: Modify the AI prompt in `write_email_from_transcript()` in `backend/bot.py` (line 116)
- **Email subject**: Change the subject line in `on_participant_left()` in `backend/bot.py` (line 107)

## Troubleshooting

- **Browser permissions**: Allow microphone access when prompted
- **Connection issues**: Try a different browser or check VPN/firewall settings
- **Audio issues**: Verify microphone and speakers are working and not muted
- **Email not sending**: Check SMTP credentials and ensure ports aren't blocked
