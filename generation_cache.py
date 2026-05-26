import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot import logger


class LocalGenerationCache:
    def __init__(self, base_dir: str | Path, config: dict):
        self.base_dir = Path(base_dir)
        self.config = config
        self.cache_dir = self.base_dir / "generation_cache"
        self.images_dir = self.cache_dir / "images"
        self.prompt_index_dir = self.cache_dir / "by_prompt"
        self.records_path = self.cache_dir / "generation_records.jsonl"
        self._lock = asyncio.Lock()

    def _enabled(self) -> bool:
        return bool(self.config.get("enable_local_generation_cache", True))

    def _history_limit(self) -> int:
        try:
            value = int(self.config.get("local_generation_prompt_history_limit", 10))
        except Exception:
            value = 10
        return max(1, value)

    def _image_ext(self, image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith(b"\xff\xd8"):
            return ".jpg"
        if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return ".gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".bin"

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256((prompt or "").strip().encode("utf-8")).hexdigest()

    def _normalize_attempts(self, attempts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not attempts:
            return []
        normalized = []
        for attempt in attempts:
            normalized.append(
                {
                    "route_name": attempt.get("route_name", ""),
                    "model": attempt.get("model", ""),
                    "api_mode": attempt.get("api_mode", ""),
                    "api_url": attempt.get("api_url", ""),
                    "api_key_masked": attempt.get("api_key_masked", ""),
                    "use_text_to_image_api": bool(attempt.get("use_text_to_image_api", False)),
                    "request_kind": attempt.get("request_kind", ""),
                    "success": bool(attempt.get("success", False)),
                    "error": attempt.get("error", ""),
                    "metrics": attempt.get("metrics", {}),
                }
            )
        return normalized

    async def record_generation(
            self,
            *,
            prompt: str,
            result: bytes | str,
            attempts: list[dict[str, Any]] | None = None,
            context: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled():
            return

        prompt_text = (prompt or "").strip()
        timestamp = datetime.now().isoformat(timespec="seconds")
        prompt_hash = self._prompt_hash(prompt_text)
        result_is_image = isinstance(result, bytes) and bool(result)

        image_rel_path = ""
        image_info: dict[str, Any] = {}

        async with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.prompt_index_dir.mkdir(parents=True, exist_ok=True)

            if result_is_image:
                ext = self._image_ext(result)
                image_name = f"{timestamp.replace(':', '').replace('-', '')}_{prompt_hash[:12]}{ext}"
                image_path = self.images_dir / image_name
                image_path.write_bytes(result)
                image_rel_path = str(image_path.relative_to(self.base_dir)).replace("\\", "/")
                image_info = {
                    "path": image_rel_path,
                    "size": len(result),
                }

            record = {
                "timestamp": timestamp,
                "prompt": prompt_text,
                "prompt_hash": prompt_hash,
                "success": result_is_image,
                "result_info": image_info if result_is_image else {"message": str(result or "")[:2000]},
                "context": context or {},
                "attempts": self._normalize_attempts(attempts),
            }

            with self.records_path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

            prompt_file = self.prompt_index_dir / f"{prompt_hash}.json"
            history_payload = {"prompt": prompt_text, "prompt_hash": prompt_hash, "records": []}
            if prompt_file.exists():
                try:
                    history_payload = json.loads(prompt_file.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("LocalGenerationCache: failed to parse prompt cache, recreate file")
            history_payload["prompt"] = prompt_text
            history_payload["prompt_hash"] = prompt_hash
            records = history_payload.get("records", [])
            records.append(record)
            history_payload["records"] = records[-self._history_limit():]
            prompt_file.write_text(
                json.dumps(history_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )