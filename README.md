# 🎵 VinCreationz Audio-Reactive Video Engine

**Current Version:** v10.3 (Production)

A powerful, all-in-one Python pipeline designed for independent artists. It automatically generates 1080p audio-reactive music videos for YouTube AND professional 3000px cover art for streaming services (DistroKid/Spotify/Apple Music) in a single click.

## ✨ Key Features

### 🎬 Video Generation (YouTube)
* **Audio Reactivity:** Detects BPM and beat drops to apply "Pulse" effects (Zoom + Brightness Flash) automatically.
* **Dynamic Visualizer:** Generates a spectrum analyzer (frequency bars) overlaid on the video.
* **Ken Burns Effect:** Smoothly pans and zooms across static artwork to create movement.
* **Lyric Support:** Automatically parses `.lrc` files to display karaoke-style lyrics.
* **Smart Watermarking:** Auto-colors the "VinCreationz" watermark (black or white) based on background brightness.
* **High Quality:** Renders in 1080p (30fps) using NVENC GPU acceleration for speed.

### 🎨 Cover Art Generator (DistroKid/Streaming)
* **Auto-Formatting:** Takes your raw background image and crops/upscales it to a perfect **3000x3000px** square.
* **Text Overlay:** Automatically pulls the **Song Title** from the filename and overlays it with a drop shadow.
* **Branding:** Adds the "VinCreationz" watermark to the cover art automatically.
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
├── test_assets/         # INPUT: Drop .mp3/.wav and images here
└── batch_renders/       # OUTPUT: Videos and Cover Art appear here