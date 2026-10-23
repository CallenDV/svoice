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
import random
import html
import ipaddress

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from urllib.parse import unquote, urlsplit
from collections import deque
from pathlib import Path

import uvicorn
import pyttsx3
from vosk import Model, KaldiRecognizer

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import (
    Response,
    RedirectResponse,
    FileResponse,
    PlainTextResponse,
    JSONResponse,
)
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

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "config.json").open("r", encoding="utf-8") as f:
    config = json.load(f)

recent_queries = deque(maxlen=10)
last_request_stats = None

options = config.get("options", {})

MAX_HISTORY_MESSAGES = max(
    0, int(options.get("max_history_messages", 20))
)
KEEP_AUDIO_FILES = options.get("keep_audio_files", False)
CONVERSATION_TIMEOUT_MINUTES = max(
    1, int(options.get("conversation_timeout_minutes", 30))
)
FFMPEG_PATH = options.get("ffmpeg_path", "ffmpeg")

MAX_AUDIO_BYTES = int(
    options.get("max_audio_bytes", 20 * 1024 * 1024)
)
MAX_REQUEST_BYTES = int(
    options.get(
        "max_request_bytes",
        MAX_AUDIO_BYTES + 1024 * 1024,
    )
)

VOSK_MODEL_PATH = config.get("speech", {}).get(
    "vosk_model_path",
    "models/vosk-model-small-en-us-0.15",
)

if not os.path.isabs(VOSK_MODEL_PATH):
    VOSK_MODEL_PATH = str(BASE_DIR / VOSK_MODEL_PATH)

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

    if len(audio_data) > MAX_AUDIO_BYTES:
        logger.warning("Audio payload exceeds configured size limit")
        return "", stats

    if vosk_model is None:
        logger.warning("Vosk model is not loaded")
        return "", stats

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]

    temp_dir = tempfile.gettempdir()
    input_file = os.path.join(
        temp_dir, f"svoice_{timestamp}_{suffix}.audio"
    )
    wav_file = os.path.join(
        temp_dir, f"svoice_{timestamp}_{suffix}.wav"
    )

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
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
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
            check=False,
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

    except subprocess.TimeoutExpired:
        logger.warning("Audio conversion timed out")
        return "", stats

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

    now = datetime.now()
    timeout = timedelta(minutes=CONVERSATION_TIMEOUT_MINUTES)

    expired = [
        conv_id
        for conv_id, last_access in conversation_last_access.items()
        if now - last_access > timeout
    ]

    for conv_id in expired:
        conversation_history.pop(conv_id, None)
        conversation_last_access.pop(conv_id, None)

        logger.info(
            "Expired conversation removed: %s",
            conv_id[:8],
        )


def get_or_create_conversation(
    dialog_state: dict | None,
) -> tuple[str, list]:
    """Get a valid existing conversation or create a new one."""

    clean_old_conversations()

    conv_id = None

    if isinstance(dialog_state, dict):
        candidate = dialog_state.get("conversation_id")

        if isinstance(candidate, str):
            try:
                conv_id = str(uuid.UUID(candidate))
            except (ValueError, AttributeError):
                conv_id = None

    if conv_id and conv_id in conversation_history:
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
    """Append a turn while keeping history within configured limits."""

    if MAX_HISTORY_MESSAGES <= 0:
        conversation_history[conv_id] = []
        conversation_last_access[conv_id] = datetime.now()
        return

    history = conversation_history.setdefault(conv_id, [])

    history.append({
        "role": "user",
        "content": user_text,
    })

    if tool_call:
        history.append({
            "role": "tool",
            "content": tool_call,
        })

    history.append({
        "role": "assistant",
        "content": assistant_text,
    })

    while len(history) > MAX_HISTORY_MESSAGES:
        history.pop(0)

        while history and history[0].get("role") != "user":
            history.pop(0)

    conversation_last_access[conv_id] = datetime.now()

def normalize_text(text: str) -> str:
    """Normalize text for predictable command matching."""

    text = unquote(str(text or ""))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\bwhats\b", "what's", text)
    text = re.sub(r"\bhows\b", "how's", text)
    text = re.sub(r"\bwhos\b", "who's", text)
    text = re.sub(r"\bthats\b", "that's", text)

    return text


def extract_command_argument(
    text: str,
    patterns: list,
) -> str | None:
    """Extract the first matching argument."""

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = match.group(1).strip(" .,!?'\"")
            return value or None

    return None
def normalize_text(text: str) -> str:
    """Normalize text for predictable command matching."""
    text = text.strip().lower()
    text = text.replace("’", "'")
    return re.sub(r"\s+", " ", text)


def extract_command_argument(text: str, patterns: list) -> str | None:
    """Extract an argument using the first matching regular expression."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?'\"")
    return None

PLEASANTRY_RESPONSES = {
    "how_are_you": [
        "I'm doing great, thanks for asking!",
        "I'm feeling fantastic and ready to assist!",
        "All systems are running smoothly. How are you?",
        "I'm doing well! It's always nice to hear from you.",
        "Couldn't be better!",
        "I'm here, alert, and ready to help. Thanks for asking!",
        "Pretty good! Just waiting for your next command.",
        "I'm doing fine. I hope you're having a wonderful day!",
    ],
    "whats_up": [
        "Not much! Just standing by to help you out.",
        "I'm all ears! What's up?",
        "Everything's running smoothly.",
        "The usual: listening and ready to assist!",
    ],
    "greeting": [
        "Hello! It's great to hear from you.",
        "Hey there!",
        "Hi!",
        "Hello again! Need any help?",
        "Hey! Your personal assistant is at your service.",
    ],
    "thanks": [
        "You're very welcome!",
        "Anytime! That's what I'm here for.",
        "Happy to help!",
        "My pleasure!",
        "No problem at all!",
    ],
    "compliment": [
        "Aw, thank you! You're pretty great yourself.",
        "That's very kind of you to say!",
        "I'll take that as a five-star review!",
        "Thanks! I'll do my best to keep impressing you.",
    ],
    "apology": [
        "No worries at all!",
        "It's completely okay. Don't worry about it.",
        "No apology needed! We're good.",
        "That's alright. How can I help?",
    ],
    "goodbye": [
        "Goodbye! Take care of yourself.",
        "See you later! I'll be here when you need me.",
        "Until next time!",
        "Take care, and have a wonderful day!",
        "Signing off for now. See you soon!",
    ],
    "good_night": [
        "Good night! Sleep well.",
        "Sweet dreams!",
        "Rest up and have a peaceful night.",
    ],
    "identity": [
        "I'm S Voice, your personal voice assistant. How can I help?",
        "You can call me S Voice! I'm here to make things easier.",
        "I'm your voice assistant, ready to help with everyday tasks.",
    ],
    "capabilities": [
        "I can open apps, set timers, change supported settings, "
        "control music, search the web, and share interesting facts.",
        "Try asking me to set a timer, play music, open an app, "
        "or tell you a random fact.",
        "I'm here to help with commands, music, timers, web searches, "
        "and a little conversation too!",
    ],
    "affirmative": [
        "Of course!",
        "Absolutely! What do you have in mind?",
        "You bet!",
        "Sure thing!",
    ],
}

TRIVIA_FACTS = [
    "Octopuses have three hearts, and two of them help pump blood "
    "through the gills.",
    "A day on Venus is longer than a year on Venus.",
    "Honey can remain edible for a very long time when properly "
    "sealed and stored.",
    "Wombats produce cube-shaped droppings.",
    "The Eiffel Tower can become slightly taller in hot weather "
    "because metal expands when heated.",
    "Bananas are berries botanically, but strawberries are not "
    "true berries.",
    "Sharks existed before trees appeared on Earth.",
    "The fingerprints of a koala are remarkably similar to those "
    "of humans.",
    "Lightning can heat the air around it to temperatures hotter "
    "than the surface of the Sun.",
    "An adult human skeleton typically has 206 bones.",
    "Some turtles can absorb oxygen through specialized tissues "
    "near their tails while underwater.",
    "The Moon is slowly moving away from Earth by a few centimeters "
    "each year.",
    "A group of flamingos is commonly called a flamboyance.",
    "Sound travels faster through water than through air.",
    "The shortest war in recorded history is commonly identified "
    "as the Anglo-Zanzibar War of 1896, which lasted less than "
    "an hour.",
    "Saturn's average density is lower than that of liquid water.",
    "There are more possible arrangements of a shuffled deck of "
    "52 cards than there are atoms in many everyday objects.",
    "The heart of a blue whale is enormous and can weigh hundreds "
    "of kilograms.",
    "Some fungi can glow in the dark through a process called "
    "bioluminescence.",
    "A cloud can weigh hundreds of tonnes, even though it floats "
    "because its water droplets are spread through a large volume "
    "of air.",
]

TRIVIA_INTROS = [
    "Here's a fun fact:",
    "Did you know?",
    "Here's a little piece of trivia for you:",
    "Ready for something interesting?",
    "Here's a fact you might enjoy:",
    "Your random fact of the moment is:",
    "Here's something fascinating:",
    "Fun fact incoming!",
]


def recognize_command(text: str) -> dict:
    """
    Recognize S Voice commands using flexible, deterministic rules.
    No language model or remote API is used.
    """
    
    original = text.strip()
    normalized = normalize_text(original)

    if not normalized:
        return {"action": "regular"}

    app_name = extract_command_argument(original, [
        r"^(?:please\s+)?(?:can you\s+)?open\s+(.+?)[?.!]*$",
        r"^(?:please\s+)?(?:can you\s+)?launch\s+(.+?)[?.!]*$",
        r"^(?:please\s+)?start\s+(?:the\s+)?app\s+(.+?)[?.!]*$",
    ])

    if app_name:
        app_name = re.sub(
            r"^(?:the\s+)?app\s+",
            "",
            app_name,
            flags=re.IGNORECASE,
        ).strip()

        return {
            "action": "open_app",
            "app_name": app_name,
        }

    timer_match = re.search(
        r"\b(?:please\s+)?(?:set|start|create|make)\s+"
        r"(?:me\s+)?(?:a\s+)?timer"
        r"(?:\s+(?:for|of|lasting))?\s+"
        r"(\d+(?:\.\d+)?\s*"
        r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?))\b",
        normalized,
    )

    if not timer_match:
        timer_match = re.search(
            r"\b(?:timer|countdown)\s+"
            r"(?:for\s+)?(\d+(?:\.\d+)?\s*"
            r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?))\b",
            normalized,
        )

    if timer_match:
        return {
            "action": "set_timer",
            "canonical_time": re.sub(
                r"\s+", " ", timer_match.group(1)
            ).strip(),
        }

    setting_match = re.search(
        r"\b(?:please\s+)?"
        r"(?:turn|switch|set|enable|disable)\s+"
        r"(?:(on|off)\s+)?"
        r"(.+?)"
        r"(?:\s+(on|off|enabled|disabled))?"
        r"[?.!]*$",
        normalized,
    )

    if setting_match:
        prefix_state = setting_match.group(1)
        setting_name = setting_match.group(2).strip()
        suffix_state = setting_match.group(3)

        state = suffix_state or prefix_state

        if state:
            setting_name = re.sub(
                r"^(?:the|my)\s+",
                "",
                setting_name,
            ).strip()

            state = state.upper()

            if state == "ENABLED":
                state = "ON"
            elif state == "DISABLED":
                state = "OFF"

            if setting_name:
                return {
                    "action": "change_setting",
                    "setting_name": setting_name,
                    "state": state,
                }

    if re.search(
        r"\b(?:start|begin|make|create)\s+"
        r"(?:a\s+)?(?:new\s+)?"
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
        r"^(?:please\s+)?search\s+"
        r"(?:(?:the\s+)?web\s+)?for\s+(.+)$",
        r"^(?:please\s+)?look\s+up\s+(.+)$",
        r"^(?:please\s+)?find\s+"
        r"(?:information|info|details|results)\s+about\s+(.+)$",
        r"^(?:please\s+)?tell\s+me\s+about\s+(.+)$",
        r"^(?:please\s+)?what\s+is\s+(.+)$",
        r"^(?:please\s+)?who\s+is\s+(.+)$",
        r"^(?:please\s+)?where\s+is\s+(.+)$",
    ])

    if query:
        return {
            "action": "web_search",
            "query": query,
        }

    if re.fullmatch(
        r"(?:please\s+)?"
        r"(?:search|web search|open search|search the web|"
        r"search online|look something up)"
        r"[?.! ]*",
        normalized,
    ):
        return {"action": "web_search_prompt"}

    music_context = re.search(
        r"\b(?:music|song|track|album|playlist|artist|"
        r"spotify|youtube|audio|player|playback)\b",
        normalized,
    )

    music_action = re.search(
        r"\b(?:play|resume|pause|stop|skip|next|previous|"
        r"continue|restart|shuffle|repeat)\b",
        normalized,
    )

    if music_action and (
        music_context
        or re.fullmatch(
            r".*\b(?:pause|stop|resume|continue|skip|"
            r"next|previous|restart)\b.*",
            normalized,
        )
    ):
        play_type = "PLAY"

        if re.search(r"\b(?:pause|hold)\b", normalized):
            play_type = "PAUSE"
        elif re.search(r"\b(?:stop|halt)\b", normalized):
            play_type = "STOP"
        elif re.search(r"\b(?:skip|next)\b", normalized):
            play_type = "NEXT"
        elif re.search(r"\b(?:previous|go back)\b", normalized):
            play_type = "PREVIOUS"
        elif re.search(r"\b(?:resume|continue)\b", normalized):
            play_type = "PLAY"

        name = extract_command_argument(original, [
            r"\bplay\s+music\s+by\s+(.+)$",
            r"\bplay\s+(?:the\s+)?song\s+(.+)$",
            r"\bplay\s+(?:the\s+)?track\s+(.+)$",
            r"\bplay\s+(.+)$",
        ])

        if name and normalize_text(name) in {
            "music",
            "some music",
            "a song",
            "music please",
            "some songs",
            "something",
            "something good",
        }:
            name = None

        return {
            "action": "play_music",
            "play_type": play_type,
            "name": name,
        }

    if re.search(
        r"\b(?:sing|singing|sing me|"
        r"sing a song|can you sing)\b",
        normalized,
    ):
        return {"action": "chatbot_sing"}

    if re.search(
        r"\b(?:trivia|fun facts?|random facts?|"
        r"interesting facts?|did you know|surprise me|"
        r"something interesting|tell me a fact)\b",
        normalized,
    ):
        return {
            "action": "fixed",
            "response": (
                f"{random.choice(TRIVIA_INTROS)} "
                f"{random.choice(TRIVIA_FACTS)}"
            ),
        }

    conversational_patterns = [
        (
            r"(?:how are you(?: doing)?|how have you been|"
            r"how are things|how is it going|how's it going|"
            r"hows it going|are you doing (?:okay|ok|well|good|fine)|"
            r"are you alright|are you all right|"
            r"are you having a good day|how do you feel)",
            "how_are_you",
        ),
        (
            r"(?:what's up|whats up|what is up|"
            r"what's happening|whats happening|"
            r"how's everything|hows everything|"
            r"what are you up to|what's going on|"
            r"whats going on)",
            "whats_up",
        ),
        (
            r"(?:who are you|what are you|"
            r"what is your name|what's your name|"
            r"whats your name|do you know who you are|"
            r"are you s voice)",
            "identity",
        ),
        (
            r"(?:what can you do|what can you help me with|"
            r"what are your features|how can you help me|"
            r"what commands do you know|"
            r"what can you help with|what are your capabilities)",
            "capabilities",
        ),
        (
            r"(?:you're awesome|youre awesome|you are awesome|"
            r"you're amazing|youre amazing|you are amazing|"
            r"you're great|youre great|you are great|"
            r"you're the best|youre the best|you are the best|"
            r"good job|well done|i like you)",
            "compliment",
        ),
        (
            r"(?:sorry|i'm sorry|im sorry|"
            r"my apologies|i apologize)",
            "apology",
        ),
        (
            r"(?:good night|goodnight|nighty night|"
            r"sleep well|sweet dreams)",
            "good_night",
        ),
        (
            r"(?:goodbye|good bye|bye|bye bye|"
            r"see you|see you later|talk to you later|"
            r"catch you later|take care)",
            "goodbye",
        ),
        (
            r"(?:hi|hello|hey|hey there|"
            r"good morning|good afternoon|good evening)",
            "greeting",
        ),
        (
            r"(?:thanks|thank you|thanks a lot|"
            r"thank you very much|much appreciated)",
            "thanks",
        ),
    ]

    for pattern, response_group in conversational_patterns:
        if re.fullmatch(pattern + r"[?.!, ]*", normalized):
            return {
                "action": "fixed",
                "response": random.choice(
                    PLEASANTRY_RESPONSES[response_group]
                ),
            }

    if re.fullmatch(
        r"(?:what time is it|"
        r"tell me the time|"
        r"what's the time|"
        r"whats the time|"
        r"current time|"
        r"what is the current time)"
        r"[?.! ]*",
        normalized,
    ):
        current_time = datetime.now().strftime(
            "%I:%M %p"
        ).lstrip("0")

        return {
            "action": "fixed",
            "response": f"It is {current_time}.",
        }

    if re.fullmatch(
        r"(?:what is the date|what's the date|"
        r"whats the date|what is today's date|"
        r"what's today's date|today's date|"
        r"what day is it|what is the day)"
        r"[?.! ]*",
        normalized,
    ):
        return {
            "action": "fixed",
            "response": datetime.now().strftime(
                "Today is %A, %B %d, %Y."
            ),
        }

    return {"action": "regular"}

LEGACY_SVOICE_HOSTS = {
    "samsungbuiasr.vlingo.com",
    "samsungbuitts.vlingo.com",
    "samsungbuilocalsearch.vlingo.com",
}

SVOICE_ROUTES = {
    "/voicepad/sr",
    "/tts",
    "/dash",
    "/dash/tools",
}

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
            if result.stdout
            else "unknown version"
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
        logger.info(
            "Loading Vosk model from %s",
            VOSK_MODEL_PATH,
        )

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

templates = Jinja2Templates(directory="templates")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

@app.middleware("http")
async def normalize_legacy_request_path(
    request: Request,
    call_next,
):
    """
    Handle legacy S Voice clients without rewriting arbitrary
    external URLs into local paths.

    The Host header identifies the virtual host the client
    intended to contact. Requests for unrelated hosts are not
    silently rewritten into S Voice endpoints.
    """
    scope = request.scope

    original_path = scope.get("path", "/")
    host_header = request.headers.get("host", "")
    host = host_header.split(":", 1)[0].strip().lower()

    decoded_path = unquote(original_path)

    if decoded_path.startswith(("http://", "https://")):
        parsed = urlsplit(decoded_path)
        destination_host = (parsed.hostname or "").lower()
        destination_path = parsed.path or "/"

        if destination_host in LEGACY_SVOICE_HOSTS:
            if destination_path in SVOICE_ROUTES:
                logger.info(
                    "Accepted legacy S Voice URL: %s%s",
                    destination_host,
                    destination_path,
                )

                scope["path"] = destination_path
                scope["raw_path"] = destination_path.encode(
                    "utf-8"
                )

                if parsed.query:
                    scope["query_string"] = parsed.query.encode(
                        "utf-8"
                    )
            else:
                logger.info(
                    "Legacy S Voice host requested unsupported "
                    "path: host=%s path=%s",
                    destination_host,
                    destination_path,
                )

        else:
            logger.debug(
                "Leaving unrelated absolute URL untouched: "
                "host=%s path=%s",
                destination_host,
                destination_path,
            )

    response = await call_next(request)
    return response

@app.get("/")
async def root():
    """Provide a simple landing page for the local server."""
    return PlainTextResponse(
        "S Voice revival server is running.\n"
        "Assistant endpoint: POST /voicepad/sr\n"
        "Text-to-speech: GET /tts?text=Hello\n"
        "Dashboard: GET /dash\n"
    )


@app.get("/health")
async def health():
    """Basic server health check."""
    return {
        "status": "ok",
        "service": "S Voice revival server",
        "speech_recognition": (
            "ready" if vosk_model is not None else "unavailable"
        ),
    }

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

    if not isinstance(dialog_state, dict):
        dialog_state = {}

    conv_id, history = get_or_create_conversation(
        dialog_state
    )

    logger.info(
        "Conversation: %s (history: %d messages)",
        conv_id[:8],
        len(history),
    )

    audio_data = extract_audio_data(body)

    transcription, audio_stats = await transcribe_audio(
        audio_data
    )

    stats.update(audio_stats)

    if not transcription:
        transcription = (extract_text(body) or "").strip()

        if transcription:
            logger.info("Text input received: %s", transcription)

    if transcription:
        recent_queries.append(transcription)
    else:
        logger.warning(
            "No audio or text data found in request"
        )

    try:
        turn = int(dialog_state.get("turn", 0)) + 1
    except (ValueError, TypeError):
        turn = 1

    new_dialog_state = {
        "turn": turn,
        "conversation_id": conv_id,
    }

    t0 = time.perf_counter()
    command = recognize_command(transcription)
    stats["command_recognition"] = (
        time.perf_counter() - t0
    )

    action = command.get("action", "regular")

    logger.info("Recognized command: %s", action)

    tool_used = None

    try:
        if action == "open_app":
            app_name = command["app_name"]

            xml = generate_open_app_response(
                transcription,
                app_name,
                new_dialog_state,
            )

            final_response_text = f"Opening {app_name}"
            tool_used = f"open_app({app_name})"

        elif action == "set_timer":
            canonical_time = command["canonical_time"]

            xml = generate_set_timer_response(
                transcription,
                canonical_time,
                new_dialog_state,
            )

            final_response_text = (
                f"Setting timer for {canonical_time}"
            )

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

            tool_used = (
                f"change_setting({setting_name}, {state})"
            )

        elif action == "record_voice":
            title = command.get("title")

            xml = generate_record_voice_response(
                transcription,
                title,
                new_dialog_state,
            )

            final_response_text = "Starting voice recording"
            tool_used = f"record_voice({title or ''})"

        elif action == "web_search":
            query = command["query"]

            xml = generate_web_search_response(
                transcription,
                query,
                new_dialog_state,
            )

            final_response_text = f"Searching for {query}"
            tool_used = f"web_search({query})"

        elif action == "web_search_prompt":
            xml = generate_web_search_prompt_response(
                transcription,
                transcription,
                new_dialog_state,
            )

            final_response_text = "Here's a search option"
            tool_used = "web_search_prompt()"

        elif action == "play_music":
            play_type = command.get("play_type", "PLAY")
            name = command.get("name")

            xml = generate_play_music_response(
                transcription,
                play_type,
                name,
                new_dialog_state,
            )

            final_response_text = (
                f"Music: {play_type}"
                + (f" {name}" if name else "")
            )

            tool_used = (
                f"play_music({play_type}, {name})"
                if name
                else f"play_music({play_type})"
            )

        elif action == "chatbot_sing":
            xml = generate_chatbot_sing_response(
                transcription,
                new_dialog_state,
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
                "I'm not sure I understood that. You can ask me "
                "to open an app, set a timer, change a setting, "
                "play music, search the web, or tell you a fact."
            )

            xml = generate_regular_response(
                transcription,
                final_response_text,
                new_dialog_state,
            )

    except Exception:
        logger.exception(
            "Failed to generate response for action %s",
            action,
        )

        final_response_text = (
            "Sorry, something went wrong while processing "
            "that request. Please try again."
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

    logger.info(
        "Request completed in %.3f seconds",
        stats["total"],
    )

    return Response(
        content=xml,
        media_type="text/xml",
    )

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

    try:
        os.remove(output_path)

        await synthesize_speech(text, output_path)

        if not os.path.isfile(output_path):
            raise RuntimeError(
                "TTS did not create an audio file"
            )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="speech.wav",
            background=None,
        )

    except HTTPException:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

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
    """Update enabled tool settings from the dashboard."""

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

@app.exception_handler(404)
async def not_found_handler(
    request: Request,
    exc: HTTPException,
):
    """Return a clear 404 without pretending every URL is S Voice."""

    path = request.url.path
    path = unquote(path);

    logger.info(
        "Route not found: method=%s path=%s host=%s",
        request.method,
        path,
        request.headers.get("host", ""),
    )

    return PlainTextResponse(
        f"Not found: {path}\n"
        "This path is not registered on the S Voice server.",
        status_code=404,
    )

if __name__ == "__main__":
    bind_host = config.get("server", {}).get(
        "bind",
        "0.0.0.0",
    )

    port = config.get("server", {}).get(
        "port",
        8000,
    )

    uvicorn.run(
        "main:app",
        host=bind_host,
        port=port,
        reload=True,
    )