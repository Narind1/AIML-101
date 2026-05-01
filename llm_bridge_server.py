"""
Local LLM Bridge Server for Call Agent (LAN only)
=================================================
Runs on your PC/laptop and receives audio/text from Android app.

Endpoints:
  POST /health
  POST /agent/text   { caller_id, text }
  POST /agent/audio  { caller_id, sample_rate, channels, bits_per_sample, audio_b64 }

How to run:
  pip install -r requirements.txt
  python llm_bridge_server.py

Then set BASE_URL in:
  android/app/src/main/java/com/minor/callagent/llm/LlmBridgeConfig.kt
"""

import base64
import importlib
import io
import os
import tempfile
import wave
from typing import Optional

from flask import Flask, jsonify, request


# -----------------------------
# Optional ASR (faster-whisper)
# -----------------------------
class OptionalAsr:
    def __init__(self) -> None:
        self.model = None
        model_name = os.getenv("ASR_MODEL", "base.en")
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(model_name, compute_type="int8")
            print(f"[ASR] Loaded faster-whisper model: {model_name}")
        except Exception as exc:
            print(f"[ASR] Disabled (faster-whisper not available): {exc}")

    def transcribe_pcm16_mono(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        if not self.model:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)

            segments, _ = self.model.transcribe(wav_path)
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            return transcript
        except Exception:
            return ""
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass


# -----------------------------
# LLM model adapter
# -----------------------------
class ModelAdapter:
    """
    Dynamic adapter for your existing model.

    Set environment variables before running server:
      LLM_PY_MODULE=<python_module_name>
      LLM_CLASS_NAME=<class_name>

    The class can expose one of these methods:
      - generate_response(text)
      - generate(text)
      - infer(text)

    Fallback is a safe echo-style placeholder.
    """

    def __init__(self) -> None:
        self.model = None
        self.module_name = os.getenv("LLM_PY_MODULE", "")
        self.class_name = os.getenv("LLM_CLASS_NAME", "")

        if self.module_name and self.class_name:
            try:
                module = importlib.import_module(self.module_name)
                cls = getattr(module, self.class_name)
                self.model = cls()
                print(f"[LLM] Loaded model: {self.module_name}.{self.class_name}")
            except Exception as exc:
                print(f"[LLM] Failed to load custom model: {exc}")

    def generate_reply(self, text: str, caller_id: str = "unknown") -> str:
        text = (text or "").strip()
        if not text:
            return "Please repeat that, I could not hear clearly."

        if self.model is None:
            return f"Noted. You said: {text}. I will pass this message to the user."

        for method_name in ("generate_response", "generate", "infer"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                result = method(text)
                return str(result).strip()

        return "I understood your message. I will inform the user."


app = Flask(__name__)
asr = OptionalAsr()
llm = ModelAdapter()


@app.post("/health")
def health():
    return jsonify({"ok": True, "service": "llm-bridge"})


@app.post("/agent/text")
def agent_text():
    data = request.get_json(force=True, silent=True) or {}
    caller_id = str(data.get("caller_id", "unknown"))
    text = str(data.get("text", "")).strip()

    reply = llm.generate_reply(text=text, caller_id=caller_id)
    return jsonify({"ok": True, "transcript": text, "reply": reply})


@app.post("/agent/audio")
def agent_audio():
    data = request.get_json(force=True, silent=True) or {}
    caller_id = str(data.get("caller_id", "unknown"))

    audio_b64 = data.get("audio_b64", "")
    sample_rate = int(data.get("sample_rate", 16000))

    if not audio_b64:
        return jsonify({"ok": False, "error": "audio_b64 missing"}), 400

    try:
        pcm_bytes = base64.b64decode(audio_b64)
    except Exception:
        return jsonify({"ok": False, "error": "invalid base64 audio"}), 400

    transcript = asr.transcribe_pcm16_mono(pcm_bytes, sample_rate=sample_rate)
    if not transcript:
        # No speech recognized in this chunk (silence/noise). Return empty reply.
        return jsonify({"ok": True, "transcript": "", "reply": ""})

    reply = llm.generate_reply(text=transcript, caller_id=caller_id)
    return jsonify({"ok": True, "transcript": transcript, "reply": reply})


@app.post("/agent/audio_file")
def agent_audio_file():
    data = request.get_json(force=True, silent=True) or {}
    caller_id = str(data.get("caller_id", "unknown"))
    wav_b64 = data.get("wav_b64", "")

    if not wav_b64:
        return jsonify({"ok": False, "error": "wav_b64 missing"}), 400

    try:
        wav_bytes = base64.b64decode(wav_b64)
    except Exception:
        return jsonify({"ok": False, "error": "invalid base64 wav"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
        tmp.write(wav_bytes)

    try:
        if asr.model is None:
            return jsonify({"ok": False, "error": "ASR unavailable on server"}), 503

        segments, _ = asr.model.transcribe(wav_path)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()

        if not transcript:
            return jsonify({"ok": True, "transcript": "", "reply": ""})

        reply = llm.generate_reply(text=transcript, caller_id=caller_id)
        return jsonify({"ok": True, "transcript": transcript, "reply": reply})
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


if __name__ == "__main__":
    host = os.getenv("BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("BRIDGE_PORT", "8000"))
    print(f"[Bridge] Running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
