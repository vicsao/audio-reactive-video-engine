# 🎵 VinCreationz Audio-Reactive Video Engine

**Current Version:** v13.0 (International / Production Suite)

A powerful, all-in-one Python pipeline designed for independent artists and Suno creators. It automatically generates **1080p audio-reactive music videos** for YouTube, **3000px square cover art** for streaming services (DistroKid/Spotify), and **1080x1920 vertical art** optimized for Suno’s mobile feed—all in a single click.

**New in v13.0:**
* **Suno "Safe-Zone" Engine:** Auto-generates vertical cover art with centered text (700px width) to survive Suno's aggressive zoom/crop.
* **Regex LRC Sanitizer:** Aggressively merges broken lyric fragments (e.g., "right (hey)") from Suno into clean, professional subtitles.
* **Smart Typography:** Automatically measures and wraps long song titles for both Square and Vertical formats to prevent text cut-offs.
* **Dev Mode:** Toggle `RENDER_VIDEO=False` in `config.txt` to instantly generate Art, Thumbnails, and clean LRCs without waiting for video rendering.

## ✨ Key Features

### 🎬 Video Generation (YouTube & Social)
* **Dual-Language Lyrics:** Automatically parses and stacks `.lrc` (Master) and `_en.lrc` (Translation) files for a professional international look.
* **Regex Lyric Sanitizer:** [NEW] Aggressively detects and merges broken Suno lyric fragments (e.g., "right (hey)") into clean, synchronized lines using advanced Regex patterns.
* **Smart Audio Reactivity:** Detects BPM and beat drops to apply "Pulse" effects (Zoom + Brightness Flash) automatically.
* **Dynamic Visualizer:** Generates a custom spectrum analyzer (frequency bars) overlaid on the video.
* **Social Optimization:** Renders in 1080p (30fps) using the `yuv420p` color space and NVENC GPU acceleration for maximum compatibility.

### 🎨 Multi-Format Art Generator
* **DistroKid Ready:** [NEW] Auto-crops and formats artwork to a perfect **3000x3000px** square, meeting strict global streaming specs.
* **Suno "Safe-Zone" Engine:** [NEW] Generates **1080x1920 (9x16)** vertical art with a 700px "safe text" column to survive Suno's aggressive mobile zoom-cropping.
* **Smart Text Wrapping:** [NEW] Intelligently measures pixel width to wrap long titles (e.g., "WE CAN GO WHEREVER SHE LIKE") without cutting off text or overlapping watermarks.
* **Unicode Branding:** Full support for Mandarin/International characters in artist names and song titles using the `Microsoft YaHei` (or custom) font stack.

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
    *(Requires: `moviepy`, `librosa`, `numpy`, `pillow`, `opencv-python`, `tqdm`)*

3.  **ImageMagick**
    Ensure [ImageMagick](https://imagemagick.org/) is installed and the path is correctly set in `main.py`.

4.  **Font Asset (Required for v13.0 Styling)**
    Place `msyhbd.ttc` (Microsoft YaHei Bold) in the project root. This is now **required** for the Suno "Safe-Zone" typography and Unicode character support.

5.  **Configuration (New)**
    Create a `config.txt` file in the root directory to control production modes (see Usage).
    ```text
    TEST_MODE=True
    RENDER_VIDEO=True
    ```

---

## 📂 Folder Structure

The script expects the following structure:

```text
/Project_Root
│
├── main.py              # The v13.0 Engine (International Production Suite)
├── config.txt           # SETTINGS: Controls RENDER_VIDEO, TEST_MODE, etc.
├── msyhbd.ttc           # MANDATORY: Microsoft YaHei Bold for Unicode/Safe-Zone Art
│
├── test_assets/         # INPUT: Drop your raw files here
│   ├── Song.wav         # Audio source
│   ├── Song.jpg         # Source Art (High Res recommended)
│   ├── Song.lrc         # Master Lyrics (Chinese/Main)
│   └── Song_en.lrc      # OPTIONAL: English Translation Lyrics
│
└── batch_renders/       # OUTPUT: The engine generates these automatically
    ├── Song_MASTER.mp4  # 1080p Music Video (YouTube)
    ├── Song_thumb.jpg   # 1080p Video Thumbnail
    ├── Song_COVERART.jpg# 3000px Square Art (DistroKid/Spotify)
    └── Song_SUNO.jpg    # 1080x1920 Vertical Art (Suno Safe-Zone)
```

---

## 🛠️ Usage

1.  **Prepare Assets:**
    * Place your audio file (`Song Name.wav` or `.mp3`) in `test_assets/`.
    * Place a matching image (`Song Name.jpg`) in `test_assets/`.
    * **Primary Lyrics:** Add `Song Name.lrc` (Mandarin/Master).
    * **Dual-Language (Optional):** Add `Song Name_en.lrc` for stacked subtitles.

2.  **Configure:**
    * Open `config.txt` to manage the production environment:
        * `TEST_MODE=True` (20s preview) or `False` (Full Song).
        * `RENDER_VIDEO=True` (Full render) or `False` (Dev Mode: Generates Art + LRCs only).
        * `ARTIST_NAME` and `WATERMARK_TEXT` (For branding).
        * `CHINESE_FONT_PATH` (Must point to `msyhbd.ttc` for Unicode).

3.  **Run the Script:**
    ```bash
    python main.py
    ```

4.  **Get Results (in `batch_renders/`):**
    * **Video:** `Song Name_MASTER.mp4` (1080p optimized for YouTube).
    * **Thumbnail:** `Song Name_thumb.jpg` (1080p video snapshot).
    * **Stream Art:** `Song Name_COVERART.jpg` (3000px Square for DistroKid).
    * **Suno Art:** `Song Name_SUNO.jpg` (9x16 Vertical for Mobile Feeds).

---

## 📄 Configuration (`config.txt`)

Manage your production settings without editing the code:

```ini
# --- PRODUCTION MODES ---
TEST_MODE=True          # True = 20s Preview. False = Full Song Render.
RENDER_VIDEO=True       # True = Make MP4. False = Generate Art/LRC/Thumb only (Dev Mode).

# --- VISUAL STYLE ---
RANDOM_CROP_IMAGES=False # True = Dynamic Camera. False = Static Art.
ARTIST_NAME=VinCreationz
WATERMARK_TEXT=VinCreationz
CHINESE_FONT_PATH=msyhbd.ttc # REQUIRED for Suno Safe-Zone & Unicode text.
```

---

**Author:** VinCreationz

