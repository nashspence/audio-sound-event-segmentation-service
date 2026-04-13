from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torchaudio
import yaml


_original_torch_load = torch.load


def _trusted_torch_load(*args: Any, **kwargs: Any):
    # ATST-SED's official checkpoints were produced before PyTorch 2.6 changed the
    # default to weights_only=True, so we restore the expected behavior here.
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


ATST_SED_REPO_DIR = Path(os.getenv("ATST_SED_REPO_DIR", "/opt/atst-sed"))
if str(ATST_SED_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(ATST_SED_REPO_DIR))

from desed_task.dataio.datasets_atst_sed import read_audio  # type: ignore  # noqa: E402
from inference import ATSTSEDInferencer  # type: ignore  # noqa: E402


class ReadyAwareATSTSEDInferencer(ATSTSEDInferencer):
    def __init__(self, *args: Any, device: torch.device, **kwargs: Any) -> None:
        self.runtime_device = device
        super().__init__(*args, **kwargs)
        self.model.to(self.runtime_device)
        self.eval()

    @torch.inference_mode()
    def forward(self, wav_file: str):  # type: ignore[override]
        mixture, _, _, _ = read_audio(wav_file, False, False, None)

        if (mixture.numel() // self.fs) <= self.audio_dur:
            inference_chunks = [mixture]
            padding_frames = 0
            mixture_pad = mixture.clone()
        else:
            mixture = mixture.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            total_chunks = (
                mixture.numel() - ((self.audio_dur - self.overlap_dur) * self.fs)
            ) // (self.overlap_dur * self.fs) + 1
            total_length = (
                total_chunks * self.overlap_dur * self.fs
                + (self.audio_dur - self.overlap_dur) * self.fs
            )
            mixture_pad = torch.nn.functional.pad(
                mixture,
                (0, 0, 0, total_length - mixture.numel()),
            )
            padding_frames = self.time2frame(total_length - mixture.numel())
            inference_chunks = self.unfolder(mixture_pad).squeeze(0).T

        sed_results = []
        for chunk in inference_chunks:
            sed_feats, atst_feats = self.feature_extractor(chunk)
            sed_feats = sed_feats.to(self.runtime_device)
            atst_feats = atst_feats.to(self.runtime_device)
            chunk_result, _ = self.model(sed_feats, atst_feats)
            sed_results.append(chunk_result)

        if self.hard_threshold is None:
            return sed_results

        chunk_decisions = []
        for chunk_result in sed_results:
            hard_chunk_result = self.post_process(chunk_result.detach().float().cpu())
            chunk_decisions.append(hard_chunk_result)
        return self.decision_unify(
            chunk_decisions,
            self.time2frame(mixture_pad.numel()),
            padding_frames,
        )


class ATSTSEDService:
    def __init__(self) -> None:
        self.service_name = "audio-sound-event-segmentation-service"
        self.model_name = "ATST-SED"
        default_base_checkpoint_url = (
            "https://drive.google.com/file/d/1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz/view?usp=drive_link"
        )
        default_stage2_checkpoint_url = (
            "https://drive.google.com/file/d/1yMv05N0Nz5mSzlQ4YBb_sqOjazPbPDhw/view?usp=sharing"
        )
        self.model_dir = Path(os.getenv("MODEL_DIR", "/models/cache"))
        self.base_checkpoint_path = Path(
            os.getenv("ATST_BASE_CHECKPOINT_PATH", str(self.model_dir / "atst_as2M.ckpt"))
        )
        self.base_checkpoint_url = os.getenv(
            "ATST_BASE_CHECKPOINT_URL",
            default_base_checkpoint_url,
        )
        self.stage2_checkpoint_path = Path(
            os.getenv(
                "ATST_SED_CHECKPOINT_PATH",
                str(self.model_dir / "Stage2_wo_ext.ckpt"),
            )
        )
        self.stage2_checkpoint_url = os.getenv(
            "ATST_SED_CHECKPOINT_URL",
            default_stage2_checkpoint_url,
        )
        self.overlap_seconds = int(float(os.getenv("ATST_SED_OVERLAP_SECONDS", "3")))
        self.hard_threshold = float(os.getenv("ATST_SED_HARD_THRESHOLD", "0.5"))
        self.excluded_labels = [
            label.strip()
            for label in os.getenv("ATST_SED_EXCLUDED_LABELS", "Speech").split(",")
            if label.strip()
        ]
        self.require_cuda = os.getenv("ATST_SED_REQUIRE_CUDA", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.runtime_config_path = self.model_dir / "stage2.inference.yaml"
        self.template_config_path = ATST_SED_REPO_DIR / "train" / "confs" / "stage2.yaml"
        self.ready = False
        self.ready_at: str | None = None
        self.device = self._resolve_device()
        self._inferencer: ReadyAwareATSTSEDInferencer | None = None
        self._lock = Lock()

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if self.require_cuda:
            raise RuntimeError("CUDA is required for this service, but no GPU is visible.")
        return torch.device("cpu")

    def _ensure_checkpoint(self, url: str, destination: Path) -> None:
        if destination.exists() and destination.stat().st_size > 0:
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial_destination = destination.with_suffix(destination.suffix + ".part")
        if partial_destination.exists():
            partial_destination.unlink()

        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                "--fuzzy",
                url,
                "-O",
                str(partial_destination),
            ],
            check=True,
        )
        partial_destination.replace(destination)

    def _write_runtime_config(self) -> None:
        with self.template_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config.setdefault("ultra", {})
        config["ultra"]["atst_init"] = str(self.base_checkpoint_path)
        config["ultra"]["model_init"] = None
        self.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runtime_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def load(self) -> None:
        if self.ready:
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_checkpoint(self.base_checkpoint_url, self.base_checkpoint_path)
        self._ensure_checkpoint(self.stage2_checkpoint_url, self.stage2_checkpoint_path)
        self._write_runtime_config()

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        inferencer = ReadyAwareATSTSEDInferencer(
            str(self.stage2_checkpoint_path),
            model_config_path=str(self.runtime_config_path),
            overlap_dur=self.overlap_seconds,
            hard_threshold=self.hard_threshold,
            device=self.device,
        )
        self._inferencer = inferencer
        self._warmup()
        self.ready = True
        self.ready_at = datetime.now(timezone.utc).isoformat()

    def _warmup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atst-sed-warmup-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            warmup_wav = tmpdir_path / "warmup.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=16000:cl=mono",
                    "-t",
                    "1",
                    str(warmup_wav),
                ],
                check=True,
            )
            self._predict_from_normalized_file(warmup_wav, "warmup.wav", include_speech=True)

    def _normalize_audio(self, source_path: Path, normalized_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(normalized_path),
            ],
            check=True,
        )

    def _predict_from_normalized_file(
        self,
        normalized_path: Path,
        filename: str,
        *,
        include_speech: bool,
    ) -> dict[str, Any]:
        if self._inferencer is None:
            raise RuntimeError("Model is not loaded.")

        with self._lock:
            decisions = self._inferencer(str(normalized_path))

        info = torchaudio.info(str(normalized_path))
        duration_seconds = 0.0
        if info.sample_rate > 0:
            duration_seconds = round(info.num_frames / info.sample_rate, 3)

        raw_segments = self._inferencer.label_encoder.decode_strong(decisions.T)
        excluded = set() if include_speech else set(self.excluded_labels)
        segments = []
        for label, start_seconds, end_seconds in raw_segments:
            if label in excluded:
                continue
            segments.append(
                {
                    "label": label,
                    "start_seconds": round(float(start_seconds), 3),
                    "end_seconds": round(float(end_seconds), 3),
                }
            )

        frame_resolution_seconds = round(
            float(
                self._inferencer.label_encoder._frame_to_time(1)
                - self._inferencer.label_encoder._frame_to_time(0)
            ),
            3,
        )
        detected_labels = sorted({segment["label"] for segment in segments})

        return {
            "model": self.model_name,
            "audio": {
                "filename": filename,
                "duration_seconds": duration_seconds,
                "sample_rate_hz": info.sample_rate,
            },
            "frame_resolution_seconds": frame_resolution_seconds,
            "excluded_labels": [] if include_speech else list(self.excluded_labels),
            "detected_labels": detected_labels,
            "segment_count": len(segments),
            "segments": segments,
        }

    def detect(self, source_path: Path, filename: str, *, include_speech: bool) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="atst-sed-request-") as tmpdir:
            normalized_path = Path(tmpdir) / "normalized.wav"
            self._normalize_audio(source_path, normalized_path)
            return self._predict_from_normalized_file(
                normalized_path,
                filename,
                include_speech=include_speech,
            )

    def health(self) -> dict[str, Any]:
        return {
            "service": self.service_name,
            "model": self.model_name,
            "ready": self.ready,
            "device": str(self.device),
            "model_loaded_at": self.ready_at,
            "excluded_labels": list(self.excluded_labels),
            "checkpoint": {
                "base": {
                    "path": str(self.base_checkpoint_path),
                    "present": self.base_checkpoint_path.exists(),
                },
                "stage2": {
                    "path": str(self.stage2_checkpoint_path),
                    "present": self.stage2_checkpoint_path.exists(),
                },
            },
        }
