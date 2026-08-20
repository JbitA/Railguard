from __future__ import annotations
from pathlib import Path
import json
import cv2
import requests


class ArtifactUploader:
    """Best-effort event-frame uploader with a local retry cache."""
    def __init__(self, api_base: str, cache_dir: str | Path = "data/event_cache", timeout_s: float = 5.0):
        self.api_base=api_base.rstrip('/')
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.timeout_s=timeout_s

    def _post(self, device_id: str, path: Path) -> str | None:
        try:
            with path.open('rb') as fh:
                r=requests.post(f"{self.api_base}/v1/events/{device_id}",files={'artifact':(path.name,fh,'image/jpeg')},timeout=self.timeout_s)
            r.raise_for_status(); data=r.json(); path.unlink(missing_ok=True); return data.get('ref')
        except Exception:
            return None

    def upload_frame(self, device_id: str, ts: str, frame_bgr) -> str | None:
        safe=ts.replace(':','-').replace('.','_')
        path=self.cache_dir/f"{device_id}_{safe}.jpg"
        ok,buf=cv2.imencode('.jpg',frame_bgr,[int(cv2.IMWRITE_JPEG_QUALITY),90])
        if not ok: return None
        path.write_bytes(buf.tobytes())
        return self._post(device_id,path)

    def flush(self, device_id: str, limit: int = 5):
        for path in sorted(self.cache_dir.glob(f"{device_id}_*.jpg"))[:limit]:
            if self._post(device_id,path) is None:
                break
