from __future__ import annotations

import base64
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

from marker.converters.pdf import PdfConverter
from marker.logger import configure_logging
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings as marker_settings


class MarkerClient:
    """Thin wrapper around the open-source `marker` SDK (local inference)."""

    def __init__(
        self,
        *,
        use_llm: bool = True,
        force_ocr: bool = False,
        torch_device: str | None = None,
        google_api_key: str | None = None,
        num_workers: int = 1,
    ) -> None:
        self._use_llm = use_llm
        self._force_ocr = force_ocr
        self._torch_device = torch_device
        self._google_api_key = google_api_key
        self._num_workers = max(1, num_workers)
        self._artifact_dict: dict[str, Any] | None = None

        configure_logging()
        self._configure_environment()

    def _configure_environment(self) -> None:
        if self._torch_device:
            os.environ["TORCH_DEVICE"] = self._torch_device
            marker_settings.TORCH_DEVICE = self._torch_device

        if self._use_llm and self._google_api_key:
            os.environ["GOOGLE_API_KEY"] = self._google_api_key
            marker_settings.GOOGLE_API_KEY = self._google_api_key

    def _get_artifact_dict(self) -> dict[str, Any]:
        if self._artifact_dict is None:
            kwargs: dict[str, Any] = {}
            if self._torch_device:
                kwargs["device"] = self._torch_device
            self._artifact_dict = create_model_dict(**kwargs)
        return self._artifact_dict

    def _build_converter(self) -> PdfConverter:
        config: dict[str, Any] = {
            "use_llm": self._use_llm,
            "force_ocr": self._force_ocr,
            "extract_images": True,
            "pdftext_workers": self._num_workers,
        }

        try:
            return PdfConverter(
                artifact_dict=self._get_artifact_dict(),
                config=config,
                renderer="marker.renderers.json.JSONRenderer",
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to initialise Marker converter: {exc}") from exc

    def _encode_images(self, images: dict[str, Any]) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for filename, image in images.items():
            buffer = io.BytesIO()
            image.save(buffer, format=marker_settings.OUTPUT_IMAGE_FORMAT)
            payload.append(
                {
                    "filename": filename,
                    "content_type": f"image/{marker_settings.OUTPUT_IMAGE_FORMAT.lower()}",
                    "data": base64.b64encode(buffer.getvalue()).decode(
                        marker_settings.OUTPUT_ENCODING
                    ),
                },
            )
        return payload

    def convert_pdf(
        self,
        file_path: Path,
    ) -> dict[str, Any]:
        """Convert PDF to structured content using Marker SDK."""
        file_path = file_path.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {file_path}")

        try:
            converter = self._build_converter()
            rendered = converter(str(file_path))
            json_text, _, images = text_from_rendered(rendered)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Marker conversion failed: {exc}") from exc

        payload: dict[str, Any] = {
            "json": json.loads(json_text),
            "images": self._encode_images(images),
            "metadata": {"json": rendered.metadata},
        }

        return {
            "request_id": f"local-{uuid.uuid4().hex}",
            "payload": payload,
        }
