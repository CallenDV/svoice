import os
import re
import json
import asyncio
import subprocess
import logging
import uuid
import time
import wave
import tempfile
import threading

from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import unquote, urlsplit
from collections import deque

import uvicorn
import pyttsx3
from vosk import Model, KaldiRecognizer

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from responsegen import (
    generate_open_app_response,
    generate_set_timer_response,
    generate_setting_change_response,
    generate_regular_response,
    generate_record_voice_response,
    generate_web_search_response,
    generate_web_search_prompt_response,
    generate_play_music_response,
    generate_chatbot_sing_response,
)

from tools import AVAILABLE_TOOLS
from multipart import extract_dialog_state, extract_audio_data, extract_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

recent_queries = deque(maxlen=10)
last_request_stats = None

options = config.get("options", {})

MAX_HISTORY_MESSAGES = options.get("max_history_messages", 20)
KEEP_AUDIO_FILES = options.get("keep_audio_files", False)
CONVERSATION_TIMEOUT_MINUTES = options.get(
    "conversation_timeout_minutes", 30
)
FFMPEG_PATH = options.get("ffmpeg_path", "ffmpeg")

VOSK_MODEL_PATH = config.get("speech", {}).get(
    "vosk_model_path",
    "models/vosk-model-small-en-us-0.15",
)

conversation_history = {}
conversation_last_access = {}

vosk_model = None
tts_lock = threading.Lock()

def transcribe_audio_sync(audio_data: bytes) -> tuple[str, dict]:
    """Convert incoming audio to text using a local Vosk model."""

    stats = {
        "save_audio": 0,
        "ffmpeg_convert": 0,
        "vosk_transcribe": 0,
    }

    if not audio_data:
        return "", stats

    if vosk_model is None:
        logger.warning("Vosk model is not loaded")
        return "", stats

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]

    input_file = f"audio_{timestamp}_{suffix}.speex"
    wav_file = f"audio_{timestamp}_{suffix}.wav"

    try:
        t0 = time.perf_counter()

        with open(input_file, "wb") as f:
            f.write(audio_data)

        stats["save_audio"] = time.perf_counter() - t0

        logger.info("Saved %d bytes of incoming audio", len(audio_data))

        t0 = time.perf_counter()

        result = subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i", input_file,
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                wav_file,
            ],
            capture_output=True,
            timeout=60,
        )

        stats["ffmpeg_convert"] = time.perf_counter() - t0

        if result.returncode != 0:
            logger.warning(
                "ffmpeg conversion failed: %s",
                result.stderr.decode(errors="replace")[-500:],
            )
            return "", stats

        t0 = time.perf_counter()

        with wave.open(wav_file, "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != 16000
            ):
                logger.warning("Unexpected WAV audio format")
                return "", stats

            recognizer = KaldiRecognizer(vosk_model, 16000)
            recognizer.SetWords(False)

            while True:
                frames = wav.readframes(4000)

                if not frames:
                    break

                recognizer.AcceptWaveform(frames)

            result = json.loads(recognizer.FinalResult())

        stats["vosk_transcribe"] = time.perf_counter() - t0

        transcription = result.get("text", "").strip()
        logger.info("Transcription: %s", transcription)

        return transcription, stats

    except Exception:
        logger.exception("Error processing incoming audio")
        return "", stats

    finally:
        if not KEEP_AUDIO_FILES:
            for path in (input_file, wav_file):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    logger.warning("Could not remove temporary file: %s", path)


async def transcribe_audio(audio_data: bytes) -> tuple[str, dict]:
    """Run local speech recognition without blocking FastAPI."""

    return await asyncio.to_thread(
        transcribe_audio_sync,
        audio_data,
    )

def synthesize_speech_sync(text: str, output_path: str):
    """Generate speech using an installed system TTS voice."""

    with tts_lock:
        engine = pyttsx3.init()

        try:
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        finally:
            engine.stop()


async def synthesize_speech(text: str, output_path: str):
    """Generate speech without blocking the async server."""

    await asyncio.to_thread(
        synthesize_speech_sync,
        text,
        output_path,
    )

def clean_old_conversations():
    """Remove conversations that have expired."""

    current_time = datetime.now()
    timeout_seconds = CONVERSATION_TIMEOUT_MINUTES * 60

    expired_convs = [
        conv_id
        for conv_id, last_access in conversation_last_access.items()
        if (current_time - last_access).total_seconds()
        > timeout_seconds
    ]

    for conv_id in expired_convs:
        conversation_history.pop(conv_id, None)
        conversation_last_access.pop(conv_id, None)

        logger.info(
            "Cleaned up expired conversation: %s",
            conv_id[:8],
        )


def get_or_create_conversation(
    dialog_state: dict | None,
) -> tuple[str, list]:
    """Get an existing conversation or create a new one."""

    clean_old_conversations()

    if dialog_state and "conversation_id" in dialog_state:
        conv_id = dialog_state["conversation_id"]

        if conv_id in conversation_history:
            conversation_last_access[conv_id] = datetime.now()
            return conv_id, conversation_history[conv_id]

    conv_id = str(uuid.uuid4())

    conversation_history[conv_id] = []
    conversation_last_access[conv_id] = datetime.now()

    return conv_id, conversation_history[conv_id]


def add_to_conversation(
    conv_id: str,
    user_text: str,
    assistant_text: str,
    tool_call: str = None,
):
    """Add a turn to conversation history."""

    if conv_id not in conversation_history:
        conversation_history[conv_id] = []

    conversation_history[conv_id].append({
        "role": "user",
        "content": user_text,
    })

    if tool_call:
        conversation_history[conv_id].append({
            "role": "tool",
            "content": tool_call,
        })

    conversation_history[conv_id].append({
        "role": "assistant",
        "content": assistant_text,
    })

    if MAX_HISTORY_MESSAGES <= 0:
        conversation_history[conv_id].clear()
    elif len(conversation_history[conv_id]) > MAX_HISTORY_MESSAGES:
        conversation_history[conv_id] = (
            conversation_history[conv_id][-MAX_HISTORY_MESSAGES:]
        )

    conversation_last_access[conv_id] = datetime.now()

def normalize_text(text: str) -> str:
    """Normalize text for predictable command matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_command_argument(text: str, patterns: list) -> str | None:
    """Extract an argument using the first matching regular expression."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?'\"")
    return None


def recognize_command(text: str) -> dict:
    """
    Match recognized commands using deterministic rules.
    No language model or remote API is used.
    """

    original = text.strip()
    normalized = normalize_text(original)

    if not normalized:
        return {"action": "regular"}

    app_name = extract_command_argument(original, [
        r"^(?:please\s+)?open\s+(.+)$",
        r"^(?:please\s+)?launch\s+(.+)$",
        r"^(?:please\s+)?start\s+app\s+(.+)$",
    ])

    if app_name:
        app_name = re.sub(
            r"^(?:the\s+)?app\s+",
            "",
            app_name,
            flags=re.IGNORECASE,
        )
        return {
            "action": "open_app",
            "app_name": app_name,
        }

    timer_match = re.search(
        r"\b(?:set|start)\s+(?:a\s+)?timer"
        r"(?:\s+(?:for|of))?\s+"
        r"(\d+(?:\.\d+)?\s*"
        r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?))\b",
        normalized,
    )

    if timer_match:
        return {
            "action": "set_timer",
            "canonical_time": timer_match.group(1),
        }

    setting_match = re.search(
        r"\b(?:turn|switch|set|enable|disable)\s+"
        r"(.+?)\s+(on|off|enabled|disabled)\b",
        normalized,
    )

    if setting_match:
        setting = setting_match.group(1).strip()
        state = setting_match.group(2).upper()

        if state == "ENABLED":
            state = "ON"
        elif state == "DISABLED":
            state = "OFF"

        return {
            "action": "change_setting",
            "setting_name": setting,
            "state": state,
        }

    if re.search(
        r"\b(?:start|begin)\s+(?:a\s+)?"
        r"(?:voice\s+)?record(?:ing)?\b",
        normalized,
    ):
        title = extract_command_argument(original, [
            r"\bcalled\s+(.+)$",
            r"\btitled\s+(.+)$",
            r"\bnamed\s+(.+)$",
        ])

        return {
            "action": "record_voice",
            "title": title,
        }

    query = extract_command_argument(original, [
        r"^(?:please\s+)?search\s+(?:the\s+web\s+)?for\s+(.+)$",
        r"^(?:please\s+)?look\s+up\s+(.+)$",
        r"^(?:please\s+)?find\s+information\s+about\s+(.+)$",
    ])

    if query:
        return {
            "action": "web_search",
            "query": query,
        }

    if re.fullmatch(
        r"(?:please\s+)?(?:search|web search|open search)"
        r"(?: the web)?",
        normalized,
    ):
        return {"action": "web_search_prompt"}

    if re.search(
        r"\b(?:play|resume|pause|stop|skip|next|previous)\b",
        normalized,
    ) and re.search(
        r"\b(?:music|song|track|album|playlist|artist|"
        r"spotify|youtube|audio|play|pause|stop|skip|"
        r"next|previous|resume)\b",
        normalized,
    ):
        play_type = "PLAY"

        if re.search(r"\b(?:pause|hold)\b", normalized):
            play_type = "PAUSE"
        elif re.search(r"\b(?:stop|halt)\b", normalized):
            play_type = "STOP"
        elif re.search(r"\b(?:skip|next)\b", normalized):
            play_type = "NEXT"
        elif re.search(r"\bprevious\b|\bgo back\b", normalized):
            play_type = "PREVIOUS"

        name = extract_command_argument(original, [
            r"\bplay\s+music\s+by\s+(.+)$",
            r"\bplay\s+(.+)$",
        ])

        if name and normalize_text(name) in {
            "music", "some music", "a song", "music please"
        }:
            name = None

        return {
            "action": "play_music",
            "play_type": play_type,
            "name": name,
        }

    if re.search(r"\b(?:sing|singing)\b", normalized):
        return {"action": "chatbot_sing"}

    if re.fullmatch(
        r"(?:hi|hello|hey|good morning|good afternoon|good evening)[!. ]*",
        normalized,
    ):
        return {
            "action": "fixed",
            "response": "Hello! How can I help?",
        }

    if re.fullmatch(
        r"(?:thanks|thank you|thanks a lot)[!. ]*",
        normalized,
    ):
        return {
            "action": "fixed",
            "response": "You're welcome!",
        }

    if re.fullmatch(
        r"(?:what time is it|tell me the time|current time)[?. ]*",
        normalized,
    ):
        current_time = datetime.now().strftime("%I:%M %p").lstrip("0")

        return {
            "action": "fixed",
            "response": f"It is {current_time}.",
        }

    if re.fullmatch(
        r"(?:what is the date|what's the date|today's date)[?. ]*",
        normalized,
    ):
        return {
            "action": "fixed",
            "response": datetime.now().strftime(
                "Today is %A, %B %d, %Y."
            ),
        }

    return {"action": "regular"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check ffmpeg and load the offline Vosk model."""
    global vosk_model

    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-version"],
            capture_output=True,
            timeout=5,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError("ffmpeg version check failed")

        version_line = (
            result.stdout.splitlines()[0]
            if result.stdout else "unknown version"
        )

        logger.info("ffmpeg found: %s", version_line)

    except Exception as exc:
        logger.exception("ffmpeg startup check failed")
        raise RuntimeError(
            f"Cannot start without ffmpeg: {exc}"
        ) from exc

    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(
            f"Vosk model directory not found: {VOSK_MODEL_PATH}. "
            "Download a Vosk model and configure its path."
        )

    try:
        logger.info("Loading Vosk model from %s", VOSK_MODEL_PATH)

        vosk_model = await asyncio.to_thread(
            Model,
            VOSK_MODEL_PATH,
        )

        logger.info("Vosk model loaded successfully")

    except Exception:
        logger.exception("Failed to load Vosk model")
        raise

    try:
        yield
    finally:
        vosk_model = None
        logger.info("Vosk model unloaded")


app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def normalize_legacy_request_path(request: Request, call_next):
    """Normalize legacy clients sending encoded absolute URLs."""
    scope = request.scope
    original_path = scope.get("path", "")
    decoded_path = unquote(original_path)

    if decoded_path.startswith(("http://", "https://")):
        parsed = urlsplit(decoded_path)

        if parsed.path.startswith("/"):
            logger.info(
                "Normalized legacy request path: %s -> %s",
                original_path,
                parsed.path,
            )
            scope["path"] = parsed.path
            scope["raw_path"] = parsed.path.encode("utf-8")

            if parsed.query:
                scope["query_string"] = parsed.query.encode("utf-8")

    return await call_next(request)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(directory="templates")

@app.post("/voicepad/sr")
async def asr_endpoint(request: Request):
    global last_request_stats

    stats = {}
    started = time.perf_counter()

    logger.info("POST /voicepad/sr")

    t0 = time.perf_counter()
    body = await request.body()
    stats["read_body"] = time.perf_counter() - t0

    dialog_state = extract_dialog_state(body)

    if dialog_state:
        logger.info("Dialog state received: %s", dialog_state)
    else:
        logger.info("No dialog state, new conversation")

    conv_id, history = get_or_create_conversation(dialog_state)

    logger.info(
        "Conversation: %s (history: %d messages)",
        conv_id[:8],
        len(history),
    )

    audio_data = extract_audio_data(body)

    transcription, audio_stats = await transcribe_audio(audio_data)
    stats.update(audio_stats)

    if not transcription:
        transcription = (extract_text(body) or "").strip()

        if transcription:
            logger.info("Text input received: %s", transcription)

    if transcription:
        recent_queries.append(transcription)
    else:
        logger.warning("No audio or text data found in request")

    turn = (
        dialog_state.get("turn", 0) + 1
        if dialog_state else 1
    )

    new_dialog_state = {
        "turn": turn,
        "conversation_id": conv_id,
    }

    t0 = time.perf_counter()
    command = recognize_command(transcription)
    stats["command_recognition"] = time.perf_counter() - t0

    action = command["action"]
    logger.info("Recognized command: %s", action)
    tool_used = None

    if action == "open_app":
        app_name = command["app_name"]

        xml = generate_open_app_response(
            transcription, app_name, new_dialog_state
        )
        final_response_text = f"Opening {app_name}"
        tool_used = f"open_app({app_name})"

    elif action == "set_timer":
        canonical_time = command["canonical_time"]

        xml = generate_set_timer_response(
            transcription, canonical_time, new_dialog_state
        )
        final_response_text = f"Setting timer for {canonical_time}"
        tool_used = f"set_timer({canonical_time})"

    elif action == "change_setting":
        setting_name = command["setting_name"]
        state = command["state"]

        xml = generate_setting_change_response(
            transcription,
            setting_name,
            state,
            new_dialog_state,
        )
        final_response_text = (
            f"Changing {setting_name} to {state}"
        )
        tool_used = f"change_setting({setting_name}, {state})"

    elif action == "record_voice":
        title = command.get("title")

        xml = generate_record_voice_response(
            transcription, title, new_dialog_state
        )
        final_response_text = "Starting voice recording"
        tool_used = f"record_voice({title or ''})"

    elif action == "web_search":
        query = command["query"]

        xml = generate_web_search_response(
            transcription, query, new_dialog_state
        )
        final_response_text = f"Searching for {query}"
        tool_used = f"web_search({query})"

    elif action == "web_search_prompt":
        xml = generate_web_search_prompt_response(
            transcription, transcription, new_dialog_state
        )
        final_response_text = "Here's a search option"
        tool_used = "web_search_prompt()"

    elif action == "play_music":
        play_type = command.get("play_type", "PLAY")
        name = command.get("name")

        xml = generate_play_music_response(
            transcription, play_type, name, new_dialog_state
        )

        final_response_text = (
            f"Music: {play_type}"
            + (f" {name}" if name else "")
        )
        tool_used = (
            f"play_music({play_type}, {name})"
            if name else f"play_music({play_type})"
        )

    elif action == "chatbot_sing":
        xml = generate_chatbot_sing_response(
            transcription, new_dialog_state
        )
        final_response_text = "Singing"
        tool_used = "chatbot_sing()"

    elif action == "fixed":
        final_response_text = command["response"]

        xml = generate_regular_response(
            transcription,
            final_response_text,
            new_dialog_state,
        )

    else:
        final_response_text = (
            "Sorry, I couldn't match that to a supported command. "
            "Try opening an app, setting a timer, changing a "
            "setting, controlling music, or searching the web."
        )

        xml = generate_regular_response(
            transcription,
            final_response_text,
            new_dialog_state,
        )

    if transcription:
        add_to_conversation(
            conv_id,
            transcription,
            final_response_text,
            tool_used,
        )

    xml = xml.replace("\n", "")

    stats["total"] = time.perf_counter() - started
    last_request_stats = stats

    logger.info("Full Response XML: %s", xml)

    return Response(content=xml, media_type="text/xml")

@app.get("/tts")
async def tts_endpoint(text: str):
    """Generate downloadable WAV speech using a local system voice."""

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is required",
        )

    if len(text) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Text must be 1000 characters or fewer",
        )

    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    os.remove(output_path)

    try:
        await synthesize_speech(text, output_path)

        if not os.path.isfile(output_path):
            raise RuntimeError("TTS did not create an audio file")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="speech.wav",
            background=None,
        )

    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)

        logger.exception("Text-to-speech failed")

        raise HTTPException(
            status_code=500,
            detail="Speech synthesis failed",
        )

@app.get("/dash")
async def dashboard(request: Request):
    """Render the dashboard page."""

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "tools": AVAILABLE_TOOLS,
            "recent_queries": list(reversed(recent_queries)),
            "stats": last_request_stats,
            "conversations": conversation_history,
        },
    )


@app.post("/dash/tools")
async def update_tools(request: Request):

    form_data = await request.form()
    enabled_tools = form_data.getlist("tools")

    for tool_name in AVAILABLE_TOOLS:
        AVAILABLE_TOOLS[tool_name]["enabled"] = (
            tool_name in enabled_tools
        )

    logger.info(
        "Updated tool settings: %s",
        [
            name
            for name, tool in AVAILABLE_TOOLS.items()
            if tool["enabled"]
        ],
    )

    return RedirectResponse(
        url="/dash",
        status_code=303,
    )

if __name__ == "__main__":
    bind_host = config.get("server", {}).get(
        "bind", "0.0.0.0"
    )
    port = config.get("server", {}).get("port", 8000)

    uvicorn.run(
        "main:app",
        host=bind_host,
        port=port,
        reload=True,
    )