import os
import argparse
import subprocess
import sys
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

# ----------------- API ----------------- #

app = FastAPI()

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
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
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
frontend_dir = os.path.join(project_root, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root_page():
    index_path = os.path.join(frontend_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ----------------- Main ----------------- #

@app.post("/start_bot")
async def start_bot(request: Request) -> JSONResponse:
    headers = {
        "Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{daily_api_url}/rooms",
            headers=headers,
            json={"properties": {}}
        )
        response.raise_for_status()
        room_data = response.json()
        room_url = room_data["url"]
        room_name = room_data["name"]

        exp_timestamp = int(time.time()) + MAX_SESSION_TIME
        
        token_response = await client.post(
            f"{daily_api_url}/meeting-tokens",
            headers=headers,
            json={
                "properties": {
                    "room_name": room_name,
                    "exp": exp_timestamp,
                    "is_owner": True
                }
            }
        )
        token_response.raise_for_status()
        bot_token = token_response.json()["token"]

        user_token_response = await client.post(
            f"{daily_api_url}/meeting-tokens",
            headers=headers,
            json={
                "properties": {
                    "room_name": room_name,
                    "exp": exp_timestamp,
                    "is_owner": False
                }
            }
        )
        user_token_response.raise_for_status()
        user_token = user_token_response.json()["token"]

    runner_dir = os.path.dirname(os.path.abspath(__file__))
    command = [
        sys.executable,
        "bot.py",
        "-u", room_url,
        "-t", bot_token,
    ]
    subprocess.Popen(command, cwd=runner_dir)

    return JSONResponse({
        "room_url": room_url,
        "token": user_token,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", type=str,
                        default=os.getenv("HOST", "0.0.0.0"), help="Host address")
    parser.add_argument("--port", type=int,
                        default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true",
                        default=False, help="Reload code on change")

    config = parser.parse_args()

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        uvicorn.run(
            "backend.bot_runner:app",
            host=config.host,
            port=config.port,
            reload=config.reload
        )
    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")