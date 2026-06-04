"""Embedding generation using google/siglip-base-patch16-384.

Generates 768-dimensional image and text embeddings using proper mean pooling.
"""

from __future__ import annotations

import io
import logging

import httpx
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

# Module-level cache for the model and processor (singleton pattern)
_model = None
_processor = None
_device = None


def _get_device() -> str:
    """Determine the best available device."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    logger.info("Using device: %s", _device)
    return _device


def load_model(model_name: str = "google/siglip-base-patch16-384") -> None:
    """Load the SigLIP model and processor (cached globally)."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return

    device = _get_device()
    logger.info("Loading model %s on %s...", model_name, device)

    _model = AutoModel.from_pretrained(model_name)
    _processor = AutoProcessor.from_pretrained(model_name)

    _model = _model.to(device)
    _model.eval()

    logger.info("Model loaded successfully.")


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean pool the token embeddings.

    If attention_mask is provided, only non-padded tokens are averaged.
    Otherwise, average over all tokens.
    """
    if attention_mask is not None:
        # Expand mask to match embedding dimensions
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts
    else:
        return token_embeddings.mean(dim=1)


async def generate_image_embedding(
    image_url: str,
    client: httpx.AsyncClient | None = None,
) -> list[float] | None:
    """Download an image from URL and generate its embedding.

    Returns a 768-dimensional vector as a list of floats, or None on failure.
    """
    if _model is None or _processor is None:
        load_model()

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        close_client = True

    try:
        resp = await client.get(image_url, timeout=30.0)
        if resp.status_code != 200:
            logger.warning("Failed to download image: HTTP %d for %s", resp.status_code, image_url)
            return None

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            logger.warning("Not an image response for %s: %s", image_url, content_type)
            return None

        image_data = resp.content
        image = Image.open(io.BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = _processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_get_device()) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.vision_model(**inputs)
            # outputs[0] = last_hidden_state, shape [1, num_patches+1, hidden_size]
            # Take the CLS token (first token) embedding -> [1, 768]
            embedding = outputs[0][:, 0, :]

        return embedding.cpu().detach().numpy().flatten().tolist()

    except Exception as e:
        logger.error("Error generating image embedding for %s: %s", image_url, e)
        return None
    finally:
        if close_client and client is not None:
            await client.aclose()


def generate_text_embedding(text: str) -> list[float] | None:
    """Generate a text embedding for the given text.

    Returns a 768-dimensional vector as a list of floats, or None on failure.
    """
    if _model is None or _processor is None:
        load_model()

    try:
        inputs = _processor(text=[text], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(_get_device()) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.text_model(**inputs)
            # outputs[0] = last_hidden_state, shape [1, seq_len, 768]
            # Mean pool over all tokens to get [1, 768]
            attention_mask = inputs.get("attention_mask")
            embedding = _mean_pool(outputs[0], attention_mask)

        return embedding.cpu().detach().numpy().flatten().tolist()

    except Exception as e:
        logger.error("Error generating text embedding: %s", e)
        return None
