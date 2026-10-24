"""Start the Maya website on the port the host gives (Render sets PORT; Hugging
Face Spaces uses 7860). One worker: demo rooms live in memory. The host's proxy
sets X-Forwarded-For, which uvicorn and the rate limits then trust."""

import os

import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("MAYA_TRUST_PROXY", "1")
    uvicorn.run("api:app", app_dir="src", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")),
                workers=1, proxy_headers=True, forwarded_allow_ips="*", log_level="info")
