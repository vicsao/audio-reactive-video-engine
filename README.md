# 🎵 VinCreationz Audio-Reactive Video Engine

**Current Version:** v13.2.3 (SolutionLand Master)

A professional-grade Python pipeline designed for independent artists, Suno creators, and YouTube producers. It automates the generation of **1080p audio-reactive music videos**, **3000px high-fidelity square cover art** (DistroKid/Spotify), and **1080x1920 vertical art** optimized for Suno’s mobile safe-zones—all in a single batch execution.

### **🚀 Core Production Features (v13.2.3)**
* **RTX 4070 NVENC Optimization:** Specialized FFmpeg parameters (`p7` preset, `CQ 20`) for ultra-fast, high-bitrate H.264 hardware encoding.
* **Rainbow Spectrum Analyzer:** [v13.2.2] High-detail, 160-bar audio visualizer with BPM-synced "Hard Hitting" rainbow effects.
* **5-Point Pan & Zoom (Ken Burns):** [v13.2.3] Randomized "camera" movement across background assets (Center, TL, TR, BL, BR) to eliminate visual repetition.
* **Clean Border Engine:** Implements a crisp, high-contrast 2px white border for foreground assets, providing a modern "photo" aesthetic.
* **Performance Telemetry:** [v13.2.1] Real-time monitoring of start/end clock times, total minutes produced, and a hardware "Throughput Gauge" (~6.25 - 10.31 it/s).
* **Regex LRC Sanitizer:** Aggressively merges broken lyric fragments and filters AI "hallucinations" (e.g., auto-removing "issance", "neuroscience").
* **Suno "Safe-Zone" Engine:** Auto-wraps text within a 650px-700px width limit to survive mobile feed cropping.

## ✨ Key Features

### 🎬 Video Generation (YouTube & Social)
* **4070 NVENC Optimized:** Custom-tuned FFmpeg parameters (`p7` preset, `CQ 20`) utilizing NVIDIA 40-series hardware acceleration for high-bitrate, ultra-fast encoding.
* **160-Bar Rainbow Viz:** [v13.2.2] A high-detail frequency spectrum analyzer featuring BPM-synced color gradients and "Hard Hitting" reactivity for bass-heavy tracks.
* **5-Point Dynamic Pan & Zoom:** [v13.2.3] Randomized "camera" movement across 18+ images, utilizing a 5-point point-of-interest engine (Center, TL, TR, BL, BR) to eliminate visual fatigue.
* **Dual-Language Lyrics:** Automatically parses and stacks `.lrc` (Master) and `_en.lrc` (Translation) files for professional bilingual subtitles.
* **Clean Border Engine:** Replaces legacy burn shadows with a crisp, 2px high-contrast white border, giving foreground assets a professional "gallery" aesthetic.
* **Regex Lyric Sanitizer:** Aggressively merges broken Suno lyric fragments and filters AI hallucinations (e.g., "neuroscience", "issance") using optimized Regex patterns.
* **Smart Audio Reactivity:** Automatically detects BPM to scale "Pulse" intensity (0.04 zoom for high-BPM vs 0.02 for chill tracks).

### 🎨 Multi-Format Art Generator
* **DistroKid Master:** [v13.2] Auto-crops and formats artwork to a perfect **3000x3000px** square (300 DPI equivalent), meeting strict global streaming specifications for Spotify and Apple Music.
* **Suno "Safe-Zone" Engine:** Generates **1080x1920 (9x16)** vertical art with a 650px-700px "safe text" column to ensure titles and branding survive Suno’s aggressive mobile feed cropping.
* **Shadow-Pop Branding:** Title and Watermark layers feature an increased 4px-6px black drop-shadow behind high-contrast white text to maintain 100% legibility on high-exposure or bright AI-generated backgrounds.
* **Smart Text Wrapping:** Intelligently measures pixel width based on precise font metrics to wrap long titles across multiple lines without cutting off text or overlapping the watermark.
* **International Font Stack:** [v13.2.3] Unified support for Unicode/Mandarin characters across all art outputs, utilizing the `LANGUAGE_FONT_PATH` (defaulting to Microsoft YaHei Bold) for consistent cross-platform rendering.

---

## 🚀 Installation & Setup

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/vicsao/audio-reactive-video-engine.git](https://github.com/vicsao/audio-reactive-video-engine.git)
    cd audio-reactive-video-engine
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Core Packages: `moviepy`, `librosa`, `numpy`, `pillow`, `opencv-python`, `tqdm`)*

3.  **FFmpeg & GPU Optimization (Required for v13.2+)**
    Ensure [FFmpeg](https://ffmpeg.org/) is installed and added to your System PATH. 
    * **NVENC Check:** To verify your 40-series GPU is ready, run:
      `ffmpeg -encoders | Select-String "nvenc"`
    * **Note:** The script uses the `p7` (highest quality) preset. If you are not on an RTX card, the script will require fallback to `libx264`.

4.  **ImageMagick (Text Rendering Engine)**
    Install [ImageMagick](https://imagemagick.org/). During installation, check the box: *"Install legacy utilities (e.g. convert)"*. 
    Verify the path in `main.py`:
    `change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})`

5.  **Language Font Asset**
    Ensure `msyhbd.ttc` (Microsoft YaHei Bold) or your preferred Unicode font is in the project root. This is **required** for the `LANGUAGE_FONT_PATH` setting to handle multi-language lyrics and branding shadows.

6.  **Environment Permissions**
    If running in PowerShell, ensure the execution policy allows script processing:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
    ```

7.  **Configuration (config.txt)**
    Edit the root `config.txt` to balance visual quality and render speed:
    ```ini
    TEST_MODE=True            # True = 20s preview; False = Full Song
    RENDER_VIDEO=True         # False = Art/LRC generation only
    RANDOM_CROP_IMAGES=True   # True = Pro Pan/Zoom (Slower); False = Speed
    ```

---

## 📂 Folder Structure

The script is optimized for local processing. The engine uses **Fuzzy Matching**; ensure the base filename of your audio matches your primary image and `.lrc` file. 

*Note: For the best visual results, drop 10-20 images starting with the same name (e.g., `SongName_1.jpg`, `SongName_2.jpg`) to feed the 5-point Pan & Zoom engine.*

```text
/Project_Root
│
├── main.py              # The v13.2.3 Engine (SolutionLand Master)
├── config.txt           # SETTINGS: Controls RANDOM_CROP_IMAGES, Telemetry, etc.
├── msyhbd.ttc           # MANDATORY: Unicode font for Safe-Zone Art & Multi-language lyrics
├── requirements.txt     # Python dependencies
│
├── test_assets/         # INPUT: Drop your raw source files here
│   ├── SongName.wav     # Audio source (WAV/MP3)
│   ├── SongName.jpg     # Primary Image (used for Art Engines)
│   ├── SongName_1.jpg   # Extra Slide (fed into the Slideshow Engine)
│   ├── SongName_2.jpg   # Extra Slide (fed into the Slideshow Engine)
│   ├── SongName.lrc     # Master Lyrics (Unicode/Primary Language)
│   └── SongName_en.lrc  # OPTIONAL: English Translation for Bilingual Stacking
│
└── batch_renders/       # OUTPUT: Generated automatically per batch
    ├── SongName_MASTER.mp4   # 1080p Video (4070 NVENC + Rainbow Viz)
    ├── SongName_thumb.jpg    # Captured at t=2.0s (Includes Clean Border & Branding)
    ├── SongName_COVERART.jpg # 3000px Square Art (DistroKid/Spotify Ready)
    └── SongName_SUNO.jpg     # 1080x1920 Vertical Art (Shadowed Safe-Zone Branding)
```

---

## 🛠️ Usage

1.  **Prepare Assets:**
    * **Audio:** Place your audio file (`SongName.wav` or `.mp3`) in `test_assets/`.
    * **Images (PRO Tip):** Place a matching image (`SongName.jpg`). To utilize the **5-Point Pan & Zoom Engine**, add additional images (e.g., `SongName_01.jpg`, `SongName_02.jpg`). The engine will automatically build a dynamic slideshow from all matching assets.
    * **Primary Lyrics:** Add `SongName.lrc`. This is the master timing file.
    * **Translation (Optional):** Add `SongName_en.lrc`. The engine will detect this and switch to **Bilingual Vertical Stack** mode (White Master / Yellow Translation).

2.  **Configure:**
    * Open `config.txt` to manage your local production environment:
        * `TEST_MODE=True`: Renders a 20s preview + all Art files (Best for rapid testing).
        * `RENDER_VIDEO=True`: Full MP4 render using **4070 NVENC**. Set to `False` for instant Art-only generation.
        * `RANDOM_CROP_IMAGES=True`: Enables the randomized Pan & Zoom camera. Set to `False` to gain ~25% render speed.
        * `LRC_OFFSET=5.0`: Adjusts lyric timing to align with the 5-second intro title sequence.

3.  **Run the Engine:**
    Ensure your terminal is in the project root and run:
    ```powershell
    python main.py
    ```

4.  **Review Telemetry & Results:**
    * **PowerShell Console:** Monitor the "Throughput Gauge" (~6.0 - 10.5 it/s) and the "Total Minutes Produced" summary.
    * **YouTube Master:** `SongName_MASTER.mp4` in `batch_renders/` (Includes Rainbow Viz).
    * **Streaming Art:** `SongName_COVERART.jpg` (3000px high-res square).
    * **Mobile Social:** `SongName_SUNO.jpg` (Vertical art with Shadow-Pop legibility).

---

## 📄 Configuration (`config.txt`)

Manage your production settings globally without editing the source code. This file is designed to balance the raw power of the RTX 4070 against your specific speed requirements.

```ini
# --- PRODUCTION MODES ---
# TEST_MODE: Toggle between rapid prototyping and final export.
# True  = Generates 20s previews (High it/s).
# False = Generates Full Length Master (Final Build).
TEST_MODE=True

# RENDER_VIDEO: The Master Video switch.
# True  = Full MP4 generation with 4070 NVENC acceleration.
# False = Dev Mode. Generates only Art Assets and sanitized LRCs (Instant).
RENDER_VIDEO=True

# --- VISUAL STYLE & PERFORMANCE ---
# RANDOM_CROP_IMAGES: Controls the "Ken Burns" 5-Point Pan & Zoom engine.
# True  = Pro-grade randomized camera movement (TL, TR, BL, BR, Center).
#         Performance Cost: ~20% slower it/s due to per-frame CPU math.
# False = Static Center-Zoom only. Maximizes render speed.
RANDOM_CROP_IMAGES=True

# --- BRANDING ---
# ARTIST_NAME: Main identifier appearing on thumbnails and intro titles.
ARTIST_NAME=VinCreationz

# WATERMARK_TEXT: Persistent ID in the bottom-right corner of the video.
WATERMARK_TEXT=VinCreationz

# --- LOCALIZATION & FONTS ---
# LANGUAGE_FONT_PATH: The Unicode/Multi-language engine path.
# REQUIRED for Mandarin lyrics and Suno "Safe-Zone" Art text stacking.
# Default: msyhbd.ttc (Microsoft YaHei Bold)
LANGUAGE_FONT_PATH=msyhbd.ttc

---

## 🎙️ Lyric Synchronization (batch_whisper.py)

For songs without pre-timed `.lrc` files, use the **Lyric-Sync v1.7** tool. This uses OpenAI Whisper (Medium Model) to align raw text to your audio.

### **Workflow:**
1.  **Prep:** Place your audio (`Song.wav`) and a raw text file of the lyrics (`Song.txt`) into `test_assets/generate_srt/`.
2.  **Run Sync:**
    ```powershell
    python batch_whisper.py
    ```
3.  **Automatic Move:** The script will sanitize the lyrics, generate the `.lrc` file, and move both the audio and the new `.lrc` into the main `test_assets/` folder for immediate rendering.

### **Surgeon Settings (v1.7):**
* **Initial Prompt:** Truncated to 500 chars to improve intro accuracy.
* **Hallucination Killer:** Automatically strips "érique", "transcription", and Suno style brackets `[Verse]`.
* **Stability:** `condition_on_previous_text=False` prevents error-cascading.

---
**Author:** VinCreationz

