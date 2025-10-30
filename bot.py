import asyncio
import os
import sys
import argparse
import smtplib
import ssl

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import LLMRunFrame, EndFrame
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.simli.video import SimliVideoService
from simli import SimliConfig
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from openai import AsyncOpenAI

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


async def main(room_url: str, token: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Alex | We Buy Houses",
            DailyParams(
                api_url=daily_api_url,
                api_key=daily_api_key,
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_is_live=True,
                video_out_width=512,
                video_out_height=512,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        # Re-enable Simli avatar
        simli_config = SimliConfig(
            apiKey=os.getenv("SIMLI_API_KEY", ""),
            faceId=os.getenv("SIMLI_FACE_ID", ""),
        )
        simli_ai = SimliVideoService(simli_config)

        messages = [
            {
                "role": "system",
                "content": "You are a friendly, conversational assistant for a 'We Buy Houses' company. Your name is Alex. Your goal is to gather initial information from a potential seller and book an appointment for them to speak with a human specialist.\n\nFollow these steps:\n1.  Start by warmly greeting them, introducing yourself, and confirming they're calling about selling a property.\n2.  Ask about the **property's condition**. (e.g., \"Does it need any repairs?\")\n3.  Ask for their **reason for selling**. (e.g., \"What's prompting you to sell?\")\n4.  Ask for their **timeline**. (e.g., \"How quickly are you looking to sell?\")\n5.  Ask for their **asking price**. (e.g., \"Do you have a price in mind?\")\n6.  Finally, if they seem like a good fit, your goal is to say: 'Great, this sounds like a property we can help with. The next step is a 10-minute call with our specialist, Jimmy. Are you free tomorrow morning or afternoon?'\n\nKeep your responses very brief, natural, and friendly. Only ask one question at a time.",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline([
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            simli_ai,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())
            
            # Get raw transcript
            transcript = "\n".join([
                f"{m.get('role', '')}: {m.get('content', '')}" 
                for m in context.messages
            ])
            
            # AI writes the email directly from transcript
            email_body = await write_email_from_transcript(transcript)
            
            # Send it
            await asyncio.to_thread(send_email, "New Lead: We Buy Houses", email_body)

        async def write_email_from_transcript(transcript):
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are writing an email about a new lead from a phone conversation. Include the property address, condition, reason for selling, timeline, asking price, meeting time if discussed, and any other relevant details. Write it in a clear, professional email format."
                    },
                    {
                        "role": "user",
                        "content": f"Write an email about this conversation:\n\n{transcript}"
                    }
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or "New lead from conversation."

        def send_email(subject: str, body: str):
            smtp_user = os.getenv("SMTP_USER", os.getenv("EMAIL_FROM", "james@akapulu.com"))
            smtp_pass = os.getenv("SMTP_PASS", "")
            smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "465"))
            to_email = os.getenv("LEAD_NOTIFY_EMAIL", "jimmybradford55@yahoo.com")
            from_email = smtp_user

            msg = f"From: {from_email}\r\nTo: {to_email}\r\nSubject: {subject}\r\n\r\n{body}"
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(from_email, [to_email], msg.encode("utf-8"))

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t))