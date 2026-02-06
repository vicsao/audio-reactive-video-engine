# 🎵 VinCreationz Audio-Reactive Video Engine

**Current Version:** v11.0 (Production / Gold Master)

A powerful, all-in-one Python pipeline designed for independent artists. It automatically generates 1080p audio-reactive music videos for YouTube AND professional 3000px cover art for streaming services (DistroKid/Spotify/Apple Music) in a single click.

## 🚀 New in v11.0
* **External Configuration:** Control settings via `config.txt` without touching code.
* **Creative Crop Engine:** Automatically turns static images into dynamic, panning visuals with blurred backgrounds.
* **Smart Text Layout:** Titles and subtitles automatically adjust position to avoid collisions.
* **Multi-line Support:** Handles long song titles and parenthetical "Remix" titles intelligently.
* **Visual Effects:** Includes "Difference" shadow blending and dynamic zoom.

## ✨ Key Features

### 🎬 Video Generation (YouTube)
* **Audio Reactivity:** Detects BPM and beat drops to apply "Pulse" effects (Zoom + Brightness Flash) automatically.
* **Dynamic Visualizer:** Generates a spectrum analyzer (frequency bars) overlaid on the video.
* **Ken Burns Effect:** Smoothly pans and zooms across static artwork to create movement.
* **Lyric Support:** Automatically parses `.lrc` files to display karaoke-style lyrics (supports 5s offset).
* **Smart Watermarking:** Auto-colors the watermark (black or white) based on background brightness.
* **High Quality:** Renders in 1080p (30fps) using NVENC GPU acceleration for speed.

### 🎨 Cover Art Generator (DistroKid/Streaming)
* **Auto-Formatting:** Takes your raw background image and crops/upscales it to a perfect **3000x3000px** square.
* **Smart Text Overlay:** Automatically handles line breaks for long titles or "(Remix)" tags and centers them perfectly.
* **Branding:** Adds the Artist Name watermark to the cover art automatically.
* **Format:** Saves as a high-quality JPEG (98% quality) ready for upload.

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

---

## 📂 Folder Structure

The script expects the following structure:

```text
/Project_Root
│
├── main.py              # The Engine
├── config.txt           # Settings File (NEW)
├── test_assets/         # INPUT: Drop .mp3/.wav and images here
└── batch_renders/       # OUTPUT: Videos and Cover Art appear here
```

---

## 🛠️ Usage

1.  **Prepare Assets:**
    * Place your audio file (`Song Name.wav` or `.mp3`) in `test_assets/`.
    * Place a matching image (`Song Name.jpg`) in `test_assets/`.
    * (Optional) Add a lyric file (`Song Name.lrc`).

2.  **Configure:**
    * Open `config.txt` to change settings without editing code:
        * `TEST_MODE=True` (20s render) or `False` (Full Song)
        * `ARTIST_NAME` and `WATERMARK_TEXT`
        * `RANDOM_CROP_IMAGES` (Dynamic Camera vs Static)

3.  **Run the Script:**
    ```bash
    python main.py
    ```

4.  **Get Results:**
    * **Video:** `batch_renders/Song Name_MASTER.mp4`
    * **Cover Art:** `batch_renders/Song Name_COVERART.jpg`

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