XOR
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

PREDICTIONG_FUEL:
![alt text](image-3.png)
![alt text](image-4.png)

````markdown
# GraphLogic

Project: local LLM integration demo (GraphLogic)

This repository contains a small demo showing how you connected a local LLM (exposed via an ngrok tunnel) to the OpenAI Python SDK. The demo code (test.py) exercises the `chat.completions.create` call against a local model named `local-rag-agent`.

> Note: test.py contained a hard-coded base URL and API key for local testing. Do NOT commit secrets to version control. That file is intentionally excluded per your request.

## What was done

- Created a Python script (test.py) that:
  - Instantiated `openai.OpenAI` client with a custom `base_url` pointed to an ngrok tunnel.
  - Passed an API key to the client (hard-coded in the demo — unsafe).
  - Added `ngrok-skip-browser-warning` header to avoid ngrok interstitials.
  - Sent a chat completion request (model: `local-rag-agent`) with a small message payload and printed the assistant response.
- Validated connectivity to the local LLM via the ngrok endpoint.

## Project layout (relevant)
- README.md — this document
- (test.py was used for the demo but is excluded from the repo per request)
- You can add other scripts, configs, or utilities here.

## Requirements

- Python 3.8+
- pip

Recommended Python packages:
- openai (official Python SDK)
- python-dotenv (optional, for env var management)

Example requirements entry:
```
openai
python-dotenv
```

## Safe setup & run (recommended)

1. Create a virtual environment:
   - python3 -m venv .venv
   - source .venv/bin/activate

2. Install dependencies:
   - pip install -r requirements.txt

3. Configure environment (do not hardcode secrets):
   - export OPENAI_API_KEY="your_api_key"
   - export LOCAL_LLM_BASE_URL="https://your-ngrok-url.ngrok-free.dev"

4. Run a safe demo script (example below) that uses env vars instead of embedded secrets:
   - python demo_safe.py

## demo_safe.py (safe example)
````python
// filepath: /home/fadher/CODING/AI_ML_Files/GraphLogic/demo_safe.py
import os
from openai import OpenAI

# Read configuration from environment variables (do not hardcode)
base_url = os.getenv("LOCAL_LLM_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not base_url or not api_key:
    raise SystemExit("Set LOCAL_LLM_BASE_URL and OPENAI_API_KEY environment variables")

client = OpenAI(base_url=base_url, api_key=api_key, default_headers={"ngrok-skip-browser-warning": "true"})

response = client.chat.completions.create(
    model="local-rag-agent",
    messages=[
        {"role": "system", "content": "You are a highly capable AI assistant."},
        {"role": "user", "content": "In two sentences, what is a keyboard?"}
    ],
    max_tokens=100,
    temperature=0.3
)

print(response.choices[0].message.content)
````

## Security and housekeeping

- Remove any committed secrets and rotate keys if they were exposed.
- Add secrets and environment files to `.gitignore`.
- Use environment variables or secrets management for credentials.

## Troubleshooting

- If you see ngrok interstitial pages, keep the header `"ngrok-skip-browser-warning": "true"`.
- Ensure the ngrok tunnel URL is reachable and matches the protocol (https).
- Verify the model name (`local-rag-agent`) is available on the local LLM service.

## Notes
- This repository documents the local LLM integration steps you performed. Replace placeholders and environment variables with your actual, secured configuration when running.
- The included demo avoids embedding secrets; do the same in production.

License: none specified.
````