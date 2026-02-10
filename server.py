import asyncio
import base64
import io
import json
import os
import re
import secrets
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nai-mcp", stateless_http=True, json_response=True)

NOVELAI_TOKEN = os.getenv("NOVELAI_TOKEN", "")
NOVELAI_ENDPOINT = os.getenv("NOVELAI_ENDPOINT", "https://image.novelai.net/ai/generate-image")
NOVELAI_MODEL = os.getenv("NOVELAI_MODEL", "nai-diffusion-4-5-full")
MAX_CONCURRENCY = max(1, int(os.getenv("NAI_MAX_CONCURRENCY", "1")))
REQUEST_TIMEOUT = float(os.getenv("NAI_REQUEST_TIMEOUT", "240"))
MAX_RETRIES = max(0, int(os.getenv("NAI_MAX_RETRIES", "5")))
PIXEL_LIMIT = max(64 * 64, int(os.getenv("NAI_PIXEL_LIMIT", "1048576")))

IMAGE_SAVE_DIR = os.getenv("NAI_IMAGE_SAVE_DIR", "")
PUBLIC_IMAGE_BASE = os.getenv("NAI_PUBLIC_IMAGE_BASE", "")
INCLUDE_BASE64_DEFAULT = os.getenv("NAI_INCLUDE_BASE64_DEFAULT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SIZE_PRESETS: Dict[str, Tuple[int, int]] = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}

REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)


def _parse_retry_after(header_value: str, fallback: float) -> float:
    if not header_value:
        return fallback
    try:
        val = float(header_value)
        if val >= 0:
            return min(val, 30.0)
    except ValueError:
        pass
    return fallback


def _round_to_64(value: int) -> int:
    if value <= 64:
        return 64
    return int(round(value / 64.0) * 64)


def _floor_to_64(value: int) -> int:
    if value <= 64:
        return 64
    return (value // 64) * 64


def _resolve_size(size: str, width: int, height: int) -> Tuple[int, int, bool]:
    from_preset = False
    if width > 0 and height > 0:
        used_w = width
        used_h = height
    else:
        key = (size or "portrait").strip().lower()
        used_w, used_h = SIZE_PRESETS.get(key, SIZE_PRESETS["portrait"])
        from_preset = True

    used_w = _round_to_64(used_w)
    used_h = _round_to_64(used_h)

    if used_w * used_h > PIXEL_LIMIT:
        scale = (PIXEL_LIMIT / float(used_w * used_h)) ** 0.5
        used_w = _floor_to_64(int(used_w * scale))
        used_h = _floor_to_64(int(used_h * scale))
        while used_w * used_h > PIXEL_LIMIT:
            if used_w >= used_h and used_w > 64:
                used_w -= 64
            elif used_h > 64:
                used_h -= 64
            else:
                break

    return max(64, used_w), max(64, used_h), from_preset


def _pos_to_center(pos: str) -> Optional[Dict[str, float]]:
    p = (pos or "").strip().upper()
    if len(p) != 2 or p[0] not in "ABCDE" or p[1] not in "12345":
        return None
    x = 0.5 + 0.2 * (ord(p[0]) - ord("C"))
    y = 0.5 + 0.2 * (ord(p[1]) - ord("3"))
    return {"x": round(x, 1), "y": round(y, 1)}


def _parse_characters_text(characters_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not characters_text:
        return out

    for raw in re.split(r"[;?]", characters_text):
        item = raw.strip()
        if not item:
            continue

        parts = re.split(r"\s*--uc:\s*", item, maxsplit=1, flags=re.IGNORECASE)
        left = parts[0].strip()
        uc = parts[1].strip() if len(parts) > 1 else ""

        prompt = left
        position = ""
        if "@" in left:
            p, pos = left.rsplit("@", 1)
            prompt = p.strip()
            position = pos.strip().upper()

        if prompt:
            out.append({"prompt": prompt, "uc": uc, "position": position})

    return out


def _normalize_characters(characters: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in characters or []:
        if not isinstance(raw, dict):
            continue

        prompt = str(raw.get("prompt", "")).strip()
        if not prompt:
            continue

        uc = str(raw.get("uc", raw.get("negative_prompt", ""))).strip()
        center: Optional[Dict[str, float]] = None

        center_raw = raw.get("center")
        if isinstance(center_raw, dict) and "x" in center_raw and "y" in center_raw:
            try:
                center = {"x": float(center_raw["x"]), "y": float(center_raw["y"])}
            except (TypeError, ValueError):
                center = None
        else:
            position = str(raw.get("position", "")).strip().upper()
            center = _pos_to_center(position)

        out.append({"prompt": prompt, "uc": uc, "center": center})
        if len(out) >= 6:
            break

    return out


def _apply_default_centers(characters: List[Dict[str, Any]]) -> None:
    for ch in characters:
        ch["center"] = {"x": 0.5, "y": 0.5}


def _extract_png(resp: httpx.Response) -> bytes:
    content = resp.content
    ctype = (resp.headers.get("content-type") or "").lower()

    if "zip" in ctype or content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith(".png")), None)
            if not name:
                raise RuntimeError("zip has no png")
            return zf.read(name)

    if "image/png" in ctype:
        return content

    data = resp.json()
    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"unsupported content-type: {ctype}")

    b64 = images[0]
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _save_png_and_get_url(png_bytes: bytes) -> str:
    if not IMAGE_SAVE_DIR or not PUBLIC_IMAGE_BASE:
        return ""

    try:
        out_dir = Path(IMAGE_SAVE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"nai_{secrets.token_hex(12)}.png"
        out_path = out_dir / name
        out_path.write_bytes(png_bytes)
        return f"{PUBLIC_IMAGE_BASE.rstrip('/')}/{name}"
    except Exception:
        return ""


async def _post_with_retries(payload: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
    attempt = 0
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        while True:
            try:
                resp = await client.post(NOVELAI_ENDPOINT, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= MAX_RETRIES:
                    raise
                delay = min(2.0 ** attempt, 10.0)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= MAX_RETRIES:
                    return resp
                fallback = min(2.0 ** attempt, 10.0)
                wait_s = _parse_retry_after(resp.headers.get("retry-after", ""), fallback)
                await asyncio.sleep(wait_s)
                attempt += 1
                continue

            return resp


@mcp.tool()
def ping() -> str:
    return "pong"


@mcp.tool()
async def generate_novelai_image(
    prompt: str,
    negative_prompt: str = "",
    characters_text: str = "",
    characters_json: str = "",
    size: str = "portrait",
    width: int = 0,
    height: int = 0,
    seed: int = -1,
    cfg: float = 5.0,
    steps: int = 28,
    sampler: str = "k_euler_ancestral",
    cfg_rescale: float = 0.0,
    model: str = "",
    include_base64: bool = INCLUDE_BASE64_DEFAULT,
) -> Dict[str, Any]:
    if not NOVELAI_TOKEN:
        return {"ok": False, "error": "NOVELAI_TOKEN is missing"}

    merged: List[Dict[str, Any]] = []
    merged.extend(_parse_characters_text(characters_text))

    if characters_json and characters_json.strip():
        try:
            parsed = json.loads(characters_json)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                merged.extend([item for item in parsed if isinstance(item, dict)])
        except Exception as e:
            return {"ok": False, "error": f"characters_json is invalid JSON: {e}"}

    chars = _normalize_characters(merged)

    used_model = model.strip() if model and model.strip() else NOVELAI_MODEL
    used_width, used_height, from_preset = _resolve_size(size, width, height)

    params: Dict[str, Any] = {
        "negative_prompt": negative_prompt,
        "width": used_width,
        "height": used_height,
        "steps": max(1, int(steps or 28)),
        "scale": float(cfg if cfg is not None else 5.0),
        "sampler": sampler or "k_euler_ancestral",
        "n_samples": 1,
    }

    if seed is not None and int(seed) >= 0:
        params["seed"] = int(seed)

    if cfg_rescale is not None:
        params["cfg_rescale"] = max(0.0, float(cfg_rescale))

    if "nai-diffusion-4" in used_model:
        has_manual_positions = bool(chars) and all(ch.get("center") is not None for ch in chars)
        if not has_manual_positions and chars:
            _apply_default_centers(chars)

        params["use_coords"] = has_manual_positions
        params["v4_prompt"] = {
            "caption": {"base_caption": prompt, "char_captions": []},
            "use_coords": has_manual_positions,
            "use_order": True,
        }
        params["v4_negative_prompt"] = {
            "caption": {"base_caption": negative_prompt, "char_captions": []}
        }

        if has_manual_positions:
            params["characterPrompts"] = []

        for ch in chars:
            char_caption: Dict[str, Any] = {"char_caption": ch["prompt"]}
            char_caption_neg: Dict[str, Any] = {"char_caption": ch["uc"]}

            center = ch.get("center")
            if center is not None:
                char_caption["centers"] = [center]
                char_caption_neg["centers"] = [center]
                if has_manual_positions:
                    params["characterPrompts"].append(
                        {"center": center, "prompt": ch["prompt"], "uc": ch["uc"]}
                    )

            params["v4_prompt"]["caption"]["char_captions"].append(char_caption)
            params["v4_negative_prompt"]["caption"]["char_captions"].append(char_caption_neg)

    payload = {
        "input": prompt,
        "model": used_model,
        "action": "generate",
        "parameters": params,
    }

    headers = {
        "Authorization": f"Bearer {NOVELAI_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }

    async with REQUEST_SEMAPHORE:
        try:
            resp = await _post_with_retries(payload, headers)
        except Exception as e:
            return {"ok": False, "error": f"request failed: {e}"}

    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": f"NovelAI HTTP {resp.status_code}",
            "body": resp.text[:500],
        }

    try:
        png_bytes = _extract_png(resp)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    image_url = _save_png_and_get_url(png_bytes)

    use_base64 = bool(include_base64) or not image_url

    result: Dict[str, Any] = {
        "ok": True,
        "mime_type": "image/png",
        "model": used_model,
        "width": used_width,
        "height": used_height,
        "size_mode": "preset" if from_preset else "custom",
    }

    if image_url:
        result["image_url"] = image_url
        result["image_markdown"] = f"![generated image]({image_url})"

    if use_base64:
        image_b64 = base64.b64encode(png_bytes).decode("ascii")
        result["image_base64"] = image_b64

    return result


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport="http", host="0.0.0.0", port=port, path="/mcp")

