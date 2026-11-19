"""Vision via /v1/chat/completions with image data URIs."""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)


async def describe_screen(
    base_url: str,
    image_bytes: bytes,
    prompt: str = "Describe what you see on screen.",
) -> str:
    """Send an image to the vision endpoint and get a text description."""
    import httpx

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "lfm3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
                "max_tokens": 512,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
