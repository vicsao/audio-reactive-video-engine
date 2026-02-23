# 🎵 VinCreationz Audio-Reactive Video Engine

**Current Version:** v13.1 (High-Performance Production Suite)

A professional-grade Python pipeline designed for independent artists, Suno creators, and YouTube producers. It automates the generation of **1080p audio-reactive music videos**, **3000px high-fidelity square cover art** (DistroKid/Spotify), and **1080x1920 vertical art** optimized for Suno’s mobile safe-zones—all in a single batch execution.

### **🚀 Core Production Features (v13.1)**
* **RTX 4070 NVENC Optimization:** [v13.1] Specialized FFmpeg parameters (`p7` preset, `CQ 20`) for ultra-fast, high-bitrate H.264 hardware encoding.
* **Clean Border Engine:** [v13.1] Implements a crisp, high-contrast 2px white border for foreground assets, replacing traditional burn shadows for a cleaner "photo" aesthetic.
* **Shadow-Pop Branding:** [v13.1] Title and Watermark layers in Art outputs now feature a 4px-6px black drop-shadow to ensure 100% legibility on bright/sunset background images.
* **Bilingual Vertical Stacking:** Support for Mandarin/English subtitle layering with intelligent font-path switching.
* **Regex LRC Sanitizer:** Aggressively merges broken lyric fragments and filters Whisper "hallucinations" (e.g., auto-removing "issance", "neuroscience").
* **Suno "Safe-Zone" Engine:** Auto-wraps text within a 650px-700px width limit to survive mobile feed cropping.

## ✨ Key Features

### 🎬 Video Generation (YouTube & Social)
* **4070 NVENC Optimized:** [v13.1] Custom-tuned FFmpeg parameters (`p7` preset, `CQ 20`) to utilize NVIDIA 40-series GPU power for high-bitrate, ultra-fast encoding.
* **Dual-Language Lyrics:** Automatically parses and stacks `.lrc` (Master) and `_en.lrc` (Translation) files for professional bilingual subtitles.
* **Clean Border Engine:** [v13.1] Replaces legacy burn shadows with a crisp, 2px high-contrast white border, giving foreground assets a professional "photo" aesthetic.
* **Regex Lyric Sanitizer:** Aggressively merges broken Suno lyric fragments and filters Whisper hallucinations (e.g., "issance") using advanced Regex patterns.
* **Smart Audio Reactivity:** Detects BPM and beat drops to apply "Pulse" effects (Zoom + Brightness Flash) automatically.
* **Dynamic Visualizer:** Generates a custom frequency spectrum analyzer overlaid on the bottom-third of the video.

### 🎨 Multi-Format Art Generator
* **DistroKid Master:** Auto-crops and formats artwork to a perfect **3000x3000px** square, meeting strict global streaming specifications.
* **Suno "Safe-Zone" Engine:** Generates **1080x1920 (9x16)** vertical art with a 650px-700px "safe text" column to ensure titles survive Suno’s mobile feed cropping.
* **Shadow-Pop Branding:** [v13.1] All generated art now features a 4px-6px black drop-shadow behind white text to maintain 100% legibility on bright/high-exposure backgrounds.
* **Smart Text Wrapping:** Intelligently measures pixel width based on font metrics to wrap long titles without cutting off text or overlapping the watermark.
* **Unicode Branding:** Full support for Mandarin/International characters in artist names and song titles using a robust font stack (defaulting to Microsoft YaHei Bold).

---

## 🚀 Installation

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/vicsao/audio-reactive-video-engine.git](https://github.com/vicsao/audio-reactive-video-engine.git)
    cd audio-reactive-video-engine
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Packages: `moviepy`, `librosa`, `numpy`, `pillow`, `opencv-python`, `tqdm`, `decorator`)*

3.  **FFmpeg & GPU Optimization (Required for v13.1)**
    Ensure [FFmpeg](https://ffmpeg.org/) is installed and added to your System PATH. 
    * **NVENC Check:** Open PowerShell and run `ffmpeg -encoders | Select-String "nvenc"`. If this returns no results, the script will default to CPU encoding (much slower).

4.  **ImageMagick**
    Ensure [ImageMagick](https://imagemagick.org/) is installed. Open `main.py` and verify the `IMAGEMAGICK_BINARY` path matches your installation (e.g., `C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe`).

5.  **Font Asset (Required for Production Styling)**
    Place `msyhbd.ttc` (Microsoft YaHei Bold) in the project root. This is **required** for the Suno "Safe-Zone" typography, branding shadows, and Unicode character support.

6.  **PowerShell Permissions**
    If running on Windows, ensure your execution policy allows the script to run:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
    ```

7.  **Configuration**
    Edit `config.txt` in the root directory to control production modes:
    ```text
    TEST_MODE=True        # True = 20s preview; False = Full Song
    RENDER_VIDEO=True     # False = Generate Art & LRCs only (instant)
    ARTIST_NAME=VinCreationz
    ```

---

## 📂 Folder Structure

The script is optimized for local processing. Ensure all filenames within `test_assets/` match exactly for the engine to pair them correctly.

```text
/Project_Root
│
├── main.py              # The v13.1 Engine (High-Performance Production Suite)
├── batch_whisper.py     # The v1.7 Lyric Sync Tool (One-off)
├── config.txt           # SETTINGS: Controls RENDER_VIDEO, TEST_MODE, ARTIST_NAME, etc.
├── msyhbd.ttc           # MANDATORY: Microsoft YaHei Bold for Unicode/Safe-Zone Art
├── requirements.txt     # Python dependencies
│
├── test_assets/         # INPUT: Drop your raw source files here
│   └── generate_srt/    # SYNC STAGING: Drop raw .wav + .txt here for Whisper
│   ├── SongName.wav     # Audio source (WAV/MP3)
│   ├── SongName.jpg     # Source Image (High-res source for Art Engines)
│   ├── SongName.lrc     # Master Lyrics (Chinese or Primary Language)
│   └── SongName_en.lrc  # OPTIONAL: English Translation for Bilingual Stacking
│
└── batch_renders/       # OUTPUT: The engine generates these automatically
    ├── SongName_MASTER.mp4  # 1080p Video (4070 NVENC Optimized)
    ├── SongName_thumb.jpg   # Video Thumbnail (Captured with Clean Border & Center Branding)
    ├── SongName_COVERART.jpg# 3000px Square Art (DistroKid/Spotify Specs)
    └── SongName_SUNO.jpg    # 1080x1920 Vertical Art (Shadowed Safe-Zone Branding)
```

---

## 🛠️ Usage

1.  **Prepare Assets:**
    * Place your audio file (`Song Name.wav` or `.mp3`) in `test_assets/`.
    * Place a matching image (`Song Name.jpg`) in `test_assets/`.
    * **Primary Lyrics:** Add `Song Name.lrc`. (Master file used for timing and Mandarin/Unicode).
    * **Translation (Optional):** Add `Song Name_en.lrc`. The engine will automatically detect this and switch to **Bilingual Vertical Stack** mode.

2.  **Configure:**
    * Open `config.txt` to manage your local production environment:
        * `TEST_MODE=True`: Renders a 20s preview + all Art files (Best for checking sync).
        * `TEST_MODE=False`: Renders the **Full Song** Master.
        * `RENDER_VIDEO=True`: Full MP4 render using **4070 NVENC**.
        * `RENDER_VIDEO=False`: **Dev Mode**. Generates all JPG Art and clean LRCs instantly.
        * `LRC_OFFSET=5.0`: Adjusts global lyric timing to account for the intro title sequence.

3.  **Run the Script:**
    Ensure your terminal is in the project root and run:
    ```powershell
    python main.py
    ```

4.  **Review Results (in `batch_renders/`):**
    * **YouTube:** `Song Name_MASTER.mp4` (High-bitrate, NVENC-encoded).
    * **Thumbnail:** `Song Name_thumb.jpg` (Captured at 2s to show Title + Artist branding).
    * **Streaming:** `Song Name_COVERART.jpg` (3000px high-res for DistroKid/Spotify).
    * **Social:** `Song Name_SUNO.jpg` (9x16 Vertical Art with Shadow-Pop legibility for mobile).

---

## 📄 Configuration (`config.txt`)

Manage your production settings globally without editing the source code. This file allows you to switch between fast testing and full-fidelity rendering.

```ini
# --- PRODUCTION MODES ---
TEST_MODE=True          # True = 20s Preview (fast). False = Full Song Render.
RENDER_VIDEO=True       # True = Make MP4. False = Generate Art/LRC/Thumb only (Dev Mode).

# --- VISUAL STYLE ---
RANDOM_CROP_IMAGES=False # True = Clean Border Engine (Blur + White Border). False = Static Art.
ARTIST_NAME=VinCreationz # Appears centered under the main song title.
WATERMARK_TEXT=VinCreationz # Appears in the bottom-right corner of the video.
CHINESE_FONT_PATH=msyhbd.ttc # REQUIRED for Suno Safe-Zone & Unicode text support.
```

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

