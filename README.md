# 🎵 VinCreationz Audio-Reactive Video Engine

**Current Version:** v11.0 (Production / Gold Master)

A powerful, all-in-one Python pipeline designed for independent artists. It automatically generates 1080p audio-reactive music videos for YouTube AND professional 3000px cover art for streaming services (DistroKid/Spotify/Apple Music) in a single click.

## 🚀 New in v12.0 (International Update)
* **Dual-Language Engine:** Automatically detects and stacks primary (Chinese) and secondary (English) lyrics for global appeal.
* **Language-Agnostic Logic:** Smart detection centers primary lyrics by default if no translation file (`_en.lrc`) is found.
* **Unicode Mastery:** Fully optimized for high-fidelity Mandarin character rendering using the `msyhbd.ttc` font pipeline.
* **LinkedIn-Ready Color Space:** Outputs in `yuv420p` to ensure perfect color accuracy and compatibility on social platforms.
* **External Configuration:** Full control over localization and render settings via `config.txt` without touching core code.
* **Creative Crop Engine:** Automatically converts static images into dynamic, panning visuals with blurred backgrounds.

## ✨ Key Features

### 🎬 Video Generation (YouTube & Social)
* **Dual-Language Lyrics:** Automatically parses and stacks `.lrc` (Master) and `_en.lrc` (Translation) files for a professional international look.
* **Intelligent Positioning:** Adjusts text Y-coordinates dynamically based on single or dual-layer modes to prevent collisions.
* **Audio Reactivity:** Detects BPM and beat drops to apply "Pulse" effects (Zoom + Brightness Flash) automatically.
* **Ken Burns Effect:** Smoothly pans and zooms across static artwork to create movement.
* **Dynamic Visualizer:** Generates a spectrum analyzer (frequency bars) overlaid on the video.
* **Social Optimization:** Renders in 1080p (30fps) using the `yuv420p` color space and NVENC GPU acceleration for maximum compatibility.

### 🎨 Cover Art Generator (DistroKid/Streaming)
* **Auto-Formatting:** Crops and upscales raw images to a perfect 3000x3000px square to meet global distribution specs.
* **Unicode Branding:** Supports Mandarin and other Unicode characters for artist names and titles directly on the artwork.
* **Smart Text Overlay:** Intelligently handles line breaks for long titles or "(Mandarin Version)" tags and centers them perfectly.
* **Branding & Quality:** Adds the Artist Name watermark automatically and saves as a high-quality JPEG (98% quality).

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

4.  **Font Asset (New for v12.0)**
    Place `msyhbd.ttc` (Microsoft YaHei Bold) in the project root to support Chinese/Unicode character rendering. (OPTIONAL)

---

## 📂 Folder Structure

The script expects the following structure:

```text
/Project_Root
│
├── main.py              # The v12.0 Engine (Dual-Language Support)
├── config.txt           # Settings: FONT_PATH, TEST_MODE, etc.
├── msyhbd.ttc           # MANDATORY: Microsoft YaHei Bold for Unicode/Chinese
│
├── test_assets/         # INPUT: Assets go here
│   ├── Song.wav         # Audio file
│   ├── Song.jpg         # Background image
│   ├── Song.lrc         # Primary Lyrics (Chinese/Main)
│   └── Song_en.lrc      # OPTIONAL: English Translation Lyrics
│
└── batch_renders/       # OUTPUT: Master Videos and 3000px Cover Art
```

---

## 🛠️ Usage

1.  **Prepare Assets:**
    * Place your audio file (`Song Name.wav` or `.mp3`) in `test_assets/`.
    * Place a matching image (`Song Name.jpg`) in `test_assets/`.
    * **Primary Lyrics:** Add `Song Name.lrc` (Mandarin/Master).
    * **Dual-Language (Optional):** Add `Song Name_en.lrc` for stacked subtitles.

2.  **Configure:**
    * Open `config.txt` to manage the render environment:
        * `TEST_MODE=True` (20s render) or `False` (Full Song).
        * `ARTIST_NAME` and `WATERMARK_TEXT`.
        * `RANDOM_CROP_IMAGES` (Dynamic Camera vs Static).
        * `CHINESE_FONT_PATH` (Point to `msyhbd.ttc` for Unicode support).

3.  **Run the Script:**
    ```bash
    python main.py
    ```

4.  **Get Results:**
    * **Video:** `batch_renders/Song Name_MASTER.mp4` (Optimized `yuv420p` for LinkedIn/YouTube).
    * **Cover Art:** `batch_renders/Song Name_COVERART.jpg` (3000px DistroKid ready).

---

## 📄 Configuration (`config.txt`)
Manage your settings without editing the script:
```ini
TEST_MODE=True
RANDOM_CROP_IMAGES=True
ARTIST_NAME=VinCreationz
WATERMARK_TEXT=VinCreationz
```

---

**Author:** VinCreationz

