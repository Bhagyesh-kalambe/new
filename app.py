"""
Multi-provider / multi-key LLM API gateway (Flask)

POST /go/v1 .. /go/v8   body {"q": "..."} -> JSON {"model", "answer"}

v1        -> NVIDIA Nemotron
v2, v3, v4 -> Gemini      (3 different API keys)
v5, v6, v7 -> Groq        (3 different API keys)
v8        -> OpenRouter

Nemotron / Groq / OpenRouter are streamed internally (some providers need
this), but the server buffers the whole thing and only replies once the
full answer is ready - the client always gets one complete JSON response.

API keys are hardcoded below - fill in your real ones. Don't push this to
a PUBLIC repo, or anyone who reads the code gets full use of your keys.
"""

import json
import requests
import os

from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# ---------------------------------------------------------------------
# API KEYS - one variable per endpoint slot, fill these ins
# ---------------------------------------------------------------------

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")

GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ---------------------------------------------------------------------
# ERROR DETAIL HELPER - surfaces the provider's real response body
# instead of a generic "Error code: 404" message.
# ---------------------------------------------------------------------

def extract_error_detail(e):
    resp = getattr(e, "response", None)  # requests.HTTPError and openai.APIStatusError both set this
    if resp is not None:
        try:
            body = resp.json()
        except Exception:
            body = getattr(resp, "text", str(e))
        return {"status_code": getattr(resp, "status_code", None), "body": body}
    if hasattr(e, "status_code"):
        return {"status_code": getattr(e, "status_code", None), "body": getattr(e, "body", str(e))}
    return {"status_code": None, "body": str(e)}


# ---------------------------------------------------------------------
# PROVIDER CALL FUNCTIONS - each takes the API key as a parameter, so
# the same function is reused across multiple endpoint slots with
# different keys.
# ---------------------------------------------------------------------

def call_nvidia_nemotron(api_key, question):
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": question}],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=True,  # called with streaming...
    )
    full_answer = []
    for chunk in completion:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            full_answer.append(chunk.choices[0].delta.content)
    return "".join(full_answer)  # ...buffered and returned as one complete string


def call_gemini(api_key, question):
    model = "gemini-3.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": question}]}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return str(result)


def call_groq(api_key, question):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": question}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_openrouter(api_key, question):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [{"role": "user", "content": question}],
        "reasoning": {"enabled": True},
        "stream": True,  # called with streaming...
    }
    full_answer = []
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
                delta = obj["choices"][0]["delta"].get("content")
                if delta:
                    full_answer.append(delta)
            except Exception:
                continue
    return "".join(full_answer)  # ...buffered and returned as one complete string


# version -> (model label, function, api_key to use)
VERSION_MAP = {
    "v1": ("nvidia-nemotron", call_nvidia_nemotron, NVIDIA_API_KEY),
    "v2": ("gemini-1", call_gemini, GEMINI_API_KEY_1),
    "v3": ("gemini-2", call_gemini, GEMINI_API_KEY_2),
    "v4": ("gemini-3", call_gemini, GEMINI_API_KEY_3),
    "v5": ("groq-1", call_groq, GROQ_API_KEY_1),
    "v6": ("groq-2", call_groq, GROQ_API_KEY_2),
    "v7": ("groq-3", call_groq, GROQ_API_KEY_3),
    "v8": ("openrouter", call_openrouter, OPENROUTER_API_KEY),
}


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


def make_endpoint(model_label, provider_fn, api_key):
    def endpoint():
        body = request.get_json(silent=True) or {}
        question = body.get("q") or body.get("question")
        if not question:
            return jsonify({"error": "Request body must include a 'q' field."}), 400
        try:
            answer = provider_fn(api_key, question)
            return jsonify({"model": model_label, "answer": answer})
        except Exception as e:
            return jsonify({"model": model_label, "error": extract_error_detail(e)}), 500
    return endpoint


for version, (label, fn, key) in VERSION_MAP.items():
    app.add_url_rule(f"/go/{version}", f"go_{version}", make_endpoint(label, fn, key), methods=["POST"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000, debug=False)
