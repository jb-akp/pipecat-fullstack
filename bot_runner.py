import os
import argparse
import subprocess
import sys
import traceback
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
import uvicorn

from dotenv import load_dotenv
load_dotenv(override=True)

import httpx

# ------------ Configuration ------------ #

MAX_SESSION_TIME = 5 * 60

REQUIRED_ENV_VARS = [
    'DAILY_API_KEY',
    'OPENAI_API_KEY',
    'ELEVENLABS_API_KEY',
    'ELEVENLABS_VOICE_ID',
]

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


# ----------------- API ----------------- #

app = FastAPI()

# Add exception handler to log errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_details = traceback.format_exc()
    print(f"ERROR: {error_details}", file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_details}
    )

# Add CORS middleware
# This is CRITICAL. It allows your website on localhost:3000
# to make requests to this server on localhost:8001.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve a minimal static frontend
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root_page():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Pipecat Bot Runner</h1>")

# ----------------- Main ----------------- #

@app.post("/start_bot")
async def start_bot(request: Request) -> JSONResponse:
    try:
        data = await request.json()
    except Exception as e:
        pass

    # 1. Create a new Daily WebRTC room
    print(f"DEBUG: Creating room with DAILY_API_KEY: {DAILY_API_KEY[:10]}...", file=sys.stderr)
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Create room
            print(f"DEBUG: POSTing to {DAILY_API_URL}/rooms", file=sys.stderr)
            response = await client.post(
                f"{DAILY_API_URL}/rooms",
                headers=headers,
                json={"properties": {}}
            )
            print(f"DEBUG: Room creation response status: {response.status_code}", file=sys.stderr)
            if response.status_code != 200:
                error_detail = response.text
                print(f"ERROR: Room creation failed: {response.status_code} - {error_detail}", file=sys.stderr)
                raise HTTPException(
                    status_code=500,
                    detail=f"Unable to provision room: {response.status_code} - {error_detail}")
            room_data = response.json()
            room_url = room_data["url"]
            room_name = room_data.get("name") or room_url.split("/")[-1]
            print(f"DEBUG: Room created: {room_url}, name: {room_name}", file=sys.stderr)
        except HTTPException as e:
            print(f"ERROR: HTTPException in room creation: {e.detail}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"ERROR: Exception in room creation: {e}", file=sys.stderr)
            print(f"ERROR: Traceback: {traceback.format_exc()}", file=sys.stderr)
            raise HTTPException(
                status_code=500,
                detail=f"Unable to provision room: {e}")

        # 2. Create bot token (owner)
        try:
            # exp should be a Unix timestamp, not a duration
            exp_timestamp = int(time.time()) + MAX_SESSION_TIME
            token_response = await client.post(
                f"{DAILY_API_URL}/meeting-tokens",
                headers=headers,
                json={
                    "properties": {
                        "room_name": room_name,
                        "exp": exp_timestamp,
                        "is_owner": True
                    }
                }
            )
            if token_response.status_code != 200:
                error_detail = token_response.text
                print(f"ERROR: Bot token creation failed: {token_response.status_code} - {error_detail}", file=sys.stderr)
                raise HTTPException(
                    status_code=500,
                    detail=f"Unable to create bot token: {token_response.status_code} - {error_detail}")
            bot_token = token_response.json()["token"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to create bot token: {e}")

        # 3. Create user token
        try:
            # exp should be a Unix timestamp, not a duration
            exp_timestamp = int(time.time()) + MAX_SESSION_TIME
            user_token_response = await client.post(
                f"{DAILY_API_URL}/meeting-tokens",
                headers=headers,
                json={
                    "properties": {
                        "room_name": room_name,
                        "exp": exp_timestamp,
                        "is_owner": False
                    }
                }
            )
            if user_token_response.status_code != 200:
                error_detail = user_token_response.text
                raise HTTPException(
                    status_code=500,
                    detail=f"Unable to create user token: {user_token_response.status_code} - {error_detail}")
            user_token = user_token_response.json()["token"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to create user token: {e}")

    # 4. Start bot worker subprocess
    try:
        runner_dir = os.path.dirname(os.path.abspath(__file__))
        command = [
            sys.executable,
            "bot.py",
            "-u", room_url,
            "-t", bot_token,
        ]
        subprocess.Popen(command, cwd=runner_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")

    return JSONResponse({
        "room_url": room_url,
        "token": user_token,
    })


if __name__ == "__main__":
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", type=str,
                        default=os.getenv("HOST", "0.0.0.0"), help="Host address")
    parser.add_argument("--port", type=int,
                        default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true",
                        default=False, help="Reload code on change")

    config = parser.parse_args()

    try:
        uvicorn.run(
            "bot_runner:app",
            host=config.host,
            port=config.port,
            reload=config.reload
        )
    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")