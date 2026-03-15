# MP4 → MP3 → 話者ラベル付き文字起こし

MP4 ファイルをドラッグ＆ドロップで受け取り、MP3 を抽出した上で日本語音声を文字起こしする GUI アプリケーションです。  
`faster-whisper` を使った文字起こしと、`pyannote.audio` を使った話者分離に対応しています。

## 主な機能

- MP4 をドラッグ＆ドロップして一括処理
- 既存の MP3 抽出を維持しつつ、TXT / SRT / JSON を保存
- `small` / `medium` の Whisper モデル切り替え
- 言語 `auto` / `ja` の切り替え
- 話者分離 ON / OFF 切り替え
- Hugging Face Token の GUI 入力
- `HF_TOKEN` / `HUGGINGFACE_TOKEN` 環境変数を優先
- 話者分離に失敗しても `SPEAKER_00` 固定で継続
- 抽出中 / 文字起こし中 / 話者分離中 / 保存中 の進捗表示

## 必要環境

| ソフトウェア | 備考 |
|---|---|
| Python | 3.10 以上推奨 |
| ffmpeg / ffprobe | PATH が通っていること |
| インターネット接続 | 初回の Whisper / pyannote モデル取得時に必要 |

> `ffmpeg -version` と `ffprobe -version` が通ることを確認してください。

## セットアップ

```powershell
pip install -r requirements.txt
```

### GPU を使う場合

`torch` は環境に応じたビルドが必要です。  
CUDA を使う場合は PyTorch 公式手順で先に適切な `torch` を導入し、その後で `pip install -r requirements.txt` を実行してください。  
GPU が利用できれば自動で `cuda`、利用できなければ `cpu` で動作します。

## Hugging Face Token の準備

話者分離を有効にする場合だけ必要です。未設定でも処理は止まらず、`SPEAKER_00` 固定で保存されます。

1. Hugging Face で Read 権限のトークンを作成する
2. `pyannote/speaker-diarization-3.1` の利用許諾に同意する
3. 以下のいずれかで token を設定する

```powershell
$env:HF_TOKEN="hf_xxx"
```

または

```powershell
$env:HUGGINGFACE_TOKEN="hf_xxx"
```

GUI の `HF Token` 欄に直接入力しても動作しますが、環境変数が設定されている場合は環境変数が優先されます。

## 起動方法

### バッチファイル

```powershell
.\run.bat
```

`run.bat` は依存を導入した上で GUI を起動します。

### 直接実行

```powershell
python converter.py
```

## 使い方

1. アプリを起動する
2. MP4 ファイルをドラッグ＆ドロップする
3. 出力先、MP3 品質、Whisper モデル、言語、出力形式を必要に応じて設定する
4. 話者分離を使う場合は Hugging Face Token を設定する
5. `処理開始` を押す

## 出力仕様

入力ファイル `sample.mp4` を処理すると、出力先に以下を保存します。

- `sample.mp3`
- `sample.txt`
- `sample.srt`
- `sample.json`

出力形式は GUI で複数選択できます。

### TXT

人間が読みやすい形式で保存します。

```text
[00:00:12.34 - 00:00:18.90] SPEAKER_01: こんにちは、今日は…
```

### SRT

字幕ソフトで扱いやすい標準的な SRT 形式です。各行の本文先頭に話者ラベルを付与します。

### JSON

機械処理向けに segment 単位で保存します。

```json
{
  "source_file": "sample.mp4",
  "audio_file": "sample.mp3",
  "model_size": "small",
  "language": "ja",
  "speaker_diarization_requested": true,
  "speaker_diarization_used": true,
  "warnings": [],
  "segments": [
    {
      "id": 0,
      "start": 12.34,
      "end": 18.9,
      "speaker": "SPEAKER_01",
      "text": "こんにちは、今日は…"
    }
  ]
}
```

## 話者分離フォールバック

以下のケースでは、処理全体は失敗させず `SPEAKER_00` 固定で保存します。

- Hugging Face Token が未設定
- `pyannote.audio` の読み込みに失敗
- 話者分離モデルの取得に失敗
- 話者区間の推定に失敗

GUI には警告を表示します。

## 長時間ファイルについて

- 1 時間超のファイルでも、MP3 抽出と Whisper セグメント処理は逐次実行します
- セグメントは必要最小限の `start / end / text / speaker` のみ保持します
- CPU 実行時は処理時間が長くなります
- 話者分離はモデル都合で重く、CPU では特に時間がかかります

## 動作確認スクリプト

GUI を使わずに短いサンプルで確認したい場合は `smoke_test.py` を使えます。

```powershell
python smoke_test.py .\sample.mp4 --language ja --format txt
```

話者分離も試す場合:

```powershell
python smoke_test.py .\sample.mp4 --language ja --format txt --format json --diarization
```

## トラブルシューティング

### `ffmpeg が見つかりません`

- FFmpeg をインストールし、`ffmpeg` と `ffprobe` を PATH に追加してください

### `faster-whisper が見つかりません`

- `pip install -r requirements.txt` を再実行してください

### 話者分離が動かない

- `HF_TOKEN` または `HUGGINGFACE_TOKEN` を確認してください
- `pyannote/speaker-diarization-3.1` の利用許諾に同意しているか確認してください
- 未設定や失敗時は `SPEAKER_00` 固定で保存されます

### 文字起こし結果が空になる

- 入力 MP4 に音声があるか確認してください
- 無音区間だけのファイルでは結果が空になることがあります

## ファイル構成

```text
mp4-to-text/
├── converter.py                # GUI 本体
├── transcription_pipeline.py   # MP3 抽出 / 文字起こし / 話者分離 / 保存
├── smoke_test.py               # 短いサンプル用の確認スクリプト
├── requirements.txt            # 依存パッケージ
├── run.bat                     # Windows 起動用バッチ
└── README.md
```
