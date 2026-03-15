from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str, float, str], None]
WarningCallback = Callable[[str], None]

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SPEAKER_PATTERN = re.compile(r"^SPEAKER_\d+$")

_WHISPER_MODEL_CLASS = None
_WHISPER_MODEL_LOADED = False
_TORCH_MODULE = None
_TORCH_LOADED = False
_PYANNOTE_PIPELINE_CLASS = None
_PYANNOTE_PIPELINE_LOADED = False


def _load_whisper_model_class():
    global _WHISPER_MODEL_CLASS, _WHISPER_MODEL_LOADED
    if not _WHISPER_MODEL_LOADED:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            WhisperModel = None
        _WHISPER_MODEL_CLASS = WhisperModel
        _WHISPER_MODEL_LOADED = True
    return _WHISPER_MODEL_CLASS


def _load_torch():
    global _TORCH_MODULE, _TORCH_LOADED
    if not _TORCH_LOADED:
        try:
            import torch
        except ImportError:
            torch = None
        _TORCH_MODULE = torch
        _TORCH_LOADED = True
    return _TORCH_MODULE


def _load_pyannote_pipeline_class():
    global _PYANNOTE_PIPELINE_CLASS, _PYANNOTE_PIPELINE_LOADED
    if not _PYANNOTE_PIPELINE_LOADED:
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            Pipeline = None
        _PYANNOTE_PIPELINE_CLASS = Pipeline
        _PYANNOTE_PIPELINE_LOADED = True
    return _PYANNOTE_PIPELINE_CLASS


def resolve_hf_token(explicit_token: str = "") -> str:
    for env_name in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return explicit_token.strip()


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def is_faster_whisper_available() -> bool:
    return _load_whisper_model_class() is not None


@dataclass(slots=True)
class ProcessingOptions:
    bitrate: str = "192"
    model_size: str = "small"
    language: str = "auto"
    diarization_enabled: bool = True
    hf_token: str = ""
    output_formats: tuple[str, ...] = ("txt",)


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"


@dataclass(slots=True)
class SpeakerSpan:
    start: float
    end: float
    speaker: str


@dataclass(slots=True)
class ProcessingResult:
    audio_path: Path
    transcript_paths: dict[str, Path]
    segments: list[TranscriptSegment]
    detected_language: str
    diarization_used: bool
    warnings: list[str] = field(default_factory=list)


class ProcessingError(RuntimeError):
    pass


class DependencyError(ProcessingError):
    pass


class ExternalToolError(ProcessingError):
    pass


class ProcessingCancelled(ProcessingError):
    pass


class DiarizationUnavailable(ProcessingError):
    pass


class MediaTranscriptionProcessor:
    def __init__(self) -> None:
        self._whisper_cache: dict[tuple[str, str, str], object] = {}
        self._diarization_cache: dict[tuple[str, str], object] = {}
        self._current_proc: subprocess.Popen[str] | None = None
        self._cancel_requested = False

    def reset_cancel(self) -> None:
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True
        proc = self._current_proc
        if proc and proc.poll() is None:
            proc.kill()

    def process_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        options: ProcessingOptions,
        progress_callback: ProgressCallback | None = None,
        warning_callback: WarningCallback | None = None,
    ) -> ProcessingResult:
        self._check_cancelled()

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.suffix.lower() != ".mp4":
            raise ProcessingError("MP4 ファイルのみ対応しています。")

        formats = tuple(dict.fromkeys(fmt.lower() for fmt in options.output_formats if fmt))
        if not formats:
            raise ProcessingError("出力形式が選択されていません。")

        base_path = output_dir / input_path.stem
        audio_path = base_path.with_suffix(".mp3")

        self._extract_mp3(
            input_path=input_path,
            output_path=audio_path,
            bitrate=options.bitrate,
            progress_callback=progress_callback,
        )
        raw_segments, detected_language = self._transcribe_audio(
            audio_path=audio_path,
            options=options,
            progress_callback=progress_callback,
        )

        diarization_used = False
        warnings: list[str] = []
        merged_segments = self._apply_single_speaker(raw_segments)

        if options.diarization_enabled:
            try:
                speaker_spans = self._diarize_audio(
                    audio_path=audio_path,
                    token=resolve_hf_token(options.hf_token),
                    progress_callback=progress_callback,
                )
            except DiarizationUnavailable as exc:
                warning = str(exc)
                warnings.append(warning)
                self._emit_warning(warning_callback, warning)
            else:
                merged_segments = self._assign_speakers(raw_segments, speaker_spans)
                diarization_used = True

        transcript_paths = self._save_outputs(
            input_path=input_path,
            audio_path=audio_path,
            base_path=base_path,
            output_formats=formats,
            model_size=options.model_size,
            detected_language=detected_language,
            diarization_requested=options.diarization_enabled,
            diarization_used=diarization_used,
            warnings=warnings,
            segments=merged_segments,
            progress_callback=progress_callback,
        )

        return ProcessingResult(
            audio_path=audio_path,
            transcript_paths=transcript_paths,
            segments=merged_segments,
            detected_language=detected_language,
            diarization_used=diarization_used,
            warnings=warnings,
        )

    def _extract_mp3(
        self,
        input_path: Path,
        output_path: Path,
        bitrate: str,
        progress_callback: ProgressCallback | None,
    ) -> None:
        self._check_cancelled()

        duration = self._get_duration(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            f"{bitrate}k",
            "-ar",
            "44100",
            "-progress",
            "pipe:2",
            str(output_path),
        ]

        stderr_tail: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except FileNotFoundError as exc:
            raise ExternalToolError(
                "ffmpeg が見つかりません。PATH に ffmpeg を追加してください。"
            ) from exc

        self._current_proc = proc

        try:
            if proc.stderr is not None:
                for raw_line in proc.stderr:
                    if self._cancel_requested:
                        proc.kill()
                        raise ProcessingCancelled("キャンセルされました。")

                    line = raw_line.strip()
                    if line:
                        stderr_tail.append(line)
                        if len(stderr_tail) > 30:
                            stderr_tail.pop(0)

                    match = re.match(r"out_time_us=(\d+)", line)
                    if not match:
                        continue

                    elapsed = int(match.group(1)) / 1_000_000
                    if duration and duration > 0:
                        progress = min(elapsed / duration, 1.0) * 35
                        detail = f"{format_clock(elapsed)} / {format_clock(duration)}"
                    else:
                        progress = 15
                        detail = f"経過 {format_clock(elapsed)}"
                    self._emit_progress(progress_callback, "extracting", progress, detail)

            proc.wait()
            self._check_cancelled()
        finally:
            self._current_proc = None

        if proc.returncode != 0:
            raise ExternalToolError(self._summarize_ffmpeg_error(stderr_tail))

        self._emit_progress(progress_callback, "extracting", 35, "MP3 抽出が完了しました")

    def _transcribe_audio(
        self,
        audio_path: Path,
        options: ProcessingOptions,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[TranscriptSegment], str]:
        self._check_cancelled()

        model = self._get_whisper_model(options.model_size)
        language = None if options.language == "auto" else options.language

        self._emit_progress(
            progress_callback,
            "transcribing",
            35,
            f"{options.model_size} モデルで文字起こしを開始します",
        )

        try:
            segment_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise ProcessingError(
                "faster-whisper による文字起こしに失敗しました。"
                f" 初回はモデルのダウンロードが必要です: {exc}"
            ) from exc

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0:
            duration = self._get_duration(audio_path) or 0.0

        segments: list[TranscriptSegment] = []
        for segment in segment_iter:
            self._check_cancelled()
            text = (segment.text or "").strip()
            if not text:
                continue

            end_time = float(segment.end)
            segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=end_time,
                    text=text,
                )
            )

            if duration > 0:
                progress = 35 + min(end_time / duration, 1.0) * 45
                detail = f"{format_clock(min(end_time, duration))} / {format_clock(duration)}"
            else:
                progress = min(35 + len(segments), 79)
                detail = f"{len(segments)} セグメント"
            self._emit_progress(progress_callback, "transcribing", progress, detail)

        if not segments:
            raise ProcessingError(
                "文字起こし結果が空でした。入力ファイルに音声が含まれているか確認してください。"
            )

        detected_language = getattr(info, "language", None) or (language or "auto")
        self._emit_progress(progress_callback, "transcribing", 80, "文字起こしが完了しました")
        return segments, detected_language

    def _diarize_audio(
        self,
        audio_path: Path,
        token: str,
        progress_callback: ProgressCallback | None,
    ) -> list[SpeakerSpan]:
        self._check_cancelled()

        if not token:
            raise DiarizationUnavailable(
                "Hugging Face Token が未設定のため、話者分離をスキップして `SPEAKER_00` で保存します。"
            )

        Pipeline = _load_pyannote_pipeline_class()
        if Pipeline is None:
            raise DiarizationUnavailable(
                "pyannote.audio が利用できないため、話者分離をスキップして `SPEAKER_00` で保存します。"
            )

        device = self._detect_device()
        cache_key = (token, device)
        if cache_key not in self._diarization_cache:
            try:
                pipeline = Pipeline.from_pretrained(
                    PYANNOTE_MODEL_ID,
                    use_auth_token=token,
                )
                torch = _load_torch()
                if torch is not None:
                    pipeline.to(torch.device(device))
            except Exception as exc:
                raise DiarizationUnavailable(
                    "話者分離モデルの取得に失敗しました。"
                    " Hugging Face Token とモデル利用許諾を確認し、`SPEAKER_00` で継続します。"
                ) from exc
            self._diarization_cache[cache_key] = pipeline

        self._emit_progress(progress_callback, "diarizing", 85, "話者区間を推定しています")

        try:
            annotation = self._diarization_cache[cache_key](str(audio_path))
        except Exception as exc:
            raise DiarizationUnavailable(
                f"話者分離に失敗したため、`SPEAKER_00` で継続します: {exc}"
            ) from exc

        self._check_cancelled()

        spans = [
            SpeakerSpan(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        if not spans:
            raise DiarizationUnavailable(
                "話者区間を検出できなかったため、`SPEAKER_00` で継続します。"
            )

        self._emit_progress(progress_callback, "diarizing", 90, "話者分離が完了しました")
        return spans

    def _save_outputs(
        self,
        input_path: Path,
        audio_path: Path,
        base_path: Path,
        output_formats: tuple[str, ...],
        model_size: str,
        detected_language: str,
        diarization_requested: bool,
        diarization_used: bool,
        warnings: list[str],
        segments: list[TranscriptSegment],
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Path]:
        saved_paths: dict[str, Path] = {}
        total = max(len(output_formats), 1)

        for index, output_format in enumerate(output_formats, start=1):
            self._check_cancelled()
            progress = 90 + ((index - 1) / total) * 10
            self._emit_progress(
                progress_callback,
                "saving",
                progress,
                f"{output_format.upper()} を保存しています",
            )

            output_path = base_path.with_suffix(f".{output_format}")
            if output_format == "txt":
                self._write_txt(
                    output_path=output_path,
                    input_path=input_path,
                    language=detected_language,
                    diarization_requested=diarization_requested,
                    diarization_used=diarization_used,
                    segments=segments,
                )
            elif output_format == "srt":
                self._write_srt(output_path=output_path, segments=segments)
            elif output_format == "json":
                self._write_json(
                    output_path=output_path,
                    input_path=input_path,
                    audio_path=audio_path,
                    model_size=model_size,
                    language=detected_language,
                    diarization_requested=diarization_requested,
                    diarization_used=diarization_used,
                    warnings=warnings,
                    segments=segments,
                )
            else:
                raise ProcessingError(f"未対応の出力形式です: {output_format}")

            saved_paths[output_format] = output_path

        self._emit_progress(progress_callback, "saving", 100, "保存が完了しました")
        return saved_paths

    def _write_txt(
        self,
        output_path: Path,
        input_path: Path,
        language: str,
        diarization_requested: bool,
        diarization_used: bool,
        segments: list[TranscriptSegment],
    ) -> None:
        diarization_state = "有効" if diarization_used else ("フォールバック" if diarization_requested else "OFF")
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"元ファイル: {input_path.name}\n")
            handle.write(f"言語: {language}\n")
            handle.write(f"話者分離: {diarization_state}\n")
            handle.write("\n")
            for segment in segments:
                handle.write(
                    f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] "
                    f"{segment.speaker}: {segment.text}\n"
                )

    def _write_srt(self, output_path: Path, segments: list[TranscriptSegment]) -> None:
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, segment in enumerate(segments, start=1):
                handle.write(f"{index}\n")
                handle.write(
                    f"{format_srt_timestamp(segment.start)} --> "
                    f"{format_srt_timestamp(segment.end)}\n"
                )
                handle.write(f"{segment.speaker}: {segment.text}\n\n")

    def _write_json(
        self,
        output_path: Path,
        input_path: Path,
        audio_path: Path,
        model_size: str,
        language: str,
        diarization_requested: bool,
        diarization_used: bool,
        warnings: list[str],
        segments: list[TranscriptSegment],
    ) -> None:
        payload = {
            "source_file": input_path.name,
            "audio_file": audio_path.name,
            "model_size": model_size,
            "language": language,
            "speaker_diarization_requested": diarization_requested,
            "speaker_diarization_used": diarization_used,
            "warnings": warnings,
            "segments": [
                {
                    "id": index,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "speaker": segment.speaker,
                    "text": segment.text,
                }
                for index, segment in enumerate(segments)
            ],
        }
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _get_whisper_model(self, model_size: str):
        WhisperModel = _load_whisper_model_class()
        if WhisperModel is None:
            raise DependencyError(
                "faster-whisper がインストールされていません。"
                " `pip install -r requirements.txt` を再実行してください。"
            )

        device = self._detect_device()
        compute_type = "float16" if device == "cuda" else "int8"
        cache_key = (model_size, device, compute_type)
        if cache_key not in self._whisper_cache:
            try:
                self._whisper_cache[cache_key] = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as exc:
                raise ProcessingError(
                    f"Whisper モデル `{model_size}` の読み込みに失敗しました。"
                    f" 初回はモデルのダウンロードが必要です: {exc}"
                ) from exc
        return self._whisper_cache[cache_key]

    def _assign_speakers(
        self,
        segments: list[TranscriptSegment],
        speaker_spans: list[SpeakerSpan],
    ) -> list[TranscriptSegment]:
        label_map: dict[str, str] = {}
        merged: list[TranscriptSegment] = []
        for segment in segments:
            raw_speaker = self._pick_best_speaker(segment, speaker_spans)
            speaker = self._normalize_speaker_label(raw_speaker, label_map)
            merged.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=speaker,
                )
            )
        return merged

    @staticmethod
    def _apply_single_speaker(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker="SPEAKER_00",
            )
            for segment in segments
        ]

    @staticmethod
    def _pick_best_speaker(segment: TranscriptSegment, speaker_spans: list[SpeakerSpan]) -> str:
        scores: dict[str, float] = {}
        for span in speaker_spans:
            overlap = min(segment.end, span.end) - max(segment.start, span.start)
            if overlap > 0:
                scores[span.speaker] = scores.get(span.speaker, 0.0) + overlap

        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]

        midpoint = (segment.start + segment.end) / 2
        nearest = min(
            speaker_spans,
            key=lambda span: min(abs(midpoint - span.start), abs(midpoint - span.end)),
        )
        return nearest.speaker

    @staticmethod
    def _normalize_speaker_label(raw_label: str, label_map: dict[str, str]) -> str:
        if SPEAKER_PATTERN.match(raw_label):
            return raw_label
        if raw_label not in label_map:
            label_map[raw_label] = f"SPEAKER_{len(label_map):02d}"
        return label_map[raw_label]

    @staticmethod
    def _get_duration(path: Path) -> float | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                check=False,
            )
        except FileNotFoundError:
            return None
        except Exception:
            return None

        try:
            return float(result.stdout.strip())
        except (TypeError, ValueError):
            return None

    def _detect_device(self) -> str:
        torch = _load_torch()
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _check_cancelled(self) -> None:
        if self._cancel_requested:
            raise ProcessingCancelled("キャンセルされました。")

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        stage: str,
        percent: float,
        detail: str,
    ) -> None:
        if callback is None:
            return
        callback(stage, max(0.0, min(percent, 100.0)), detail)

    @staticmethod
    def _emit_warning(callback: WarningCallback | None, message: str) -> None:
        if callback is None:
            return
        callback(message)

    @staticmethod
    def _summarize_ffmpeg_error(stderr_tail: list[str]) -> str:
        joined = " ".join(stderr_tail)
        if "Output file #0 does not contain any stream" in joined:
            return "音声トラックが見つからないため MP3 を抽出できませんでした。"
        if "No such file or directory" in joined:
            return "入力ファイルまたは出力先が見つかりません。"
        if "Permission denied" in joined:
            return "出力ファイルを書き込めません。保存先の権限を確認してください。"
        detail = stderr_tail[-1] if stderr_tail else "詳細不明"
        return f"ffmpeg による MP3 抽出に失敗しました: {detail}"


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    total_centiseconds = max(0, int(round(seconds * 100)))
    hours = total_centiseconds // 360000
    minutes = (total_centiseconds % 360000) // 6000
    secs = (total_centiseconds % 6000) // 100
    centiseconds = total_centiseconds % 100
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours = total_milliseconds // 3600000
    minutes = (total_milliseconds % 3600000) // 60000
    secs = (total_milliseconds % 60000) // 1000
    milliseconds = total_milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
