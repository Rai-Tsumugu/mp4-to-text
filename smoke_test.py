from __future__ import annotations

import argparse
from pathlib import Path

from transcription_pipeline import (
    MediaTranscriptionProcessor,
    ProcessingCancelled,
    ProcessingError,
    ProcessingOptions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MP4 -> MP3 -> text pipeline smoke test",
    )
    parser.add_argument("input_file", help="テストする MP4 ファイル")
    parser.add_argument(
        "--output-dir",
        default="smoke-output",
        help="出力先ディレクトリ",
    )
    parser.add_argument(
        "--model",
        choices=("small", "medium"),
        default="small",
        help="faster-whisper のモデルサイズ",
    )
    parser.add_argument(
        "--language",
        choices=("auto", "ja"),
        default="ja",
        help="文字起こし言語",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("txt", "srt", "json"),
        help="出力形式。省略時は txt",
    )
    parser.add_argument(
        "--bitrate",
        default="192",
        help="MP3 抽出ビットレート (kbps)",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="話者分離を有効にする",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Hugging Face Token。環境変数が設定されていればそちらが優先されます",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[error] 入力ファイルが見つかりません: {input_path}")
        return 1

    options = ProcessingOptions(
        bitrate=args.bitrate,
        model_size=args.model,
        language=args.language,
        diarization_enabled=args.diarization,
        hf_token=args.hf_token,
        output_formats=tuple(args.formats or ["txt"]),
    )
    processor = MediaTranscriptionProcessor()

    def on_progress(stage: str, percent: float, detail: str):
        suffix = f" | {detail}" if detail else ""
        print(f"[{percent:6.2f}%] {stage}{suffix}")

    def on_warning(message: str):
        print(f"[warn] {message}")

    try:
        result = processor.process_file(
            input_path=input_path,
            output_dir=args.output_dir,
            options=options,
            progress_callback=on_progress,
            warning_callback=on_warning,
        )
    except ProcessingCancelled:
        print("[cancelled] 処理がキャンセルされました")
        return 1
    except ProcessingError as exc:
        print(f"[error] {exc}")
        return 1

    print(f"[ok] MP3: {result.audio_path}")
    for output_format, path in result.transcript_paths.items():
        print(f"[ok] {output_format.upper()}: {path}")
    print(f"[ok] language={result.detected_language}, diarization_used={result.diarization_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
