"""
==============================================================================
SCRIPT: VinCreationz Shorts Generator (v1.0)
PURPOSE: Auto-converts your 1080p Master Video into a viral 9:16 Short.
LOGIC:   1. Finds loudest 60s (Chorus).
         2. Crops Master Video to 1080x1080 Square (keeps Lyrics/Viz).
         3. Stacks on blurred Album Art background.
==============================================================================
"""

import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, TextClip, ColorClip, AudioFileClip
import moviepy.video.fx.all as vfx
from PIL import Image, ImageFilter

# --- CONFIGURATION (Global) ---
INPUT_FOLDER = "test_assets"
OUTPUT_FOLDER = "batch_renders"
FONT_PATH = "msyhbd.ttc"
WATERMARK = "VinCreationz"
# ------------------------------

def find_chorus_window(audio_path, duration_limit=59):
    """Finds the 60s window with highest RMS energy."""
    print(f"    [Analyzing Audio] Finding the hook...")
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        frames_per_sec = sr / hop_length
        window_size = int(duration_limit * frames_per_sec)
        
        if len(rms) < window_size:
            return 0, librosa.get_duration(y=y, sr=sr)
            
        # Find max energy window
        window_energy = np.convolve(rms, np.ones(window_size), mode='valid')
        max_idx = np.argmax(window_energy)
        
        start = max_idx / frames_per_sec
        end = start + duration_limit
        print(f"    [Chorus Found] {int(start)}s to {int(end)}s")
        return start, end
    except Exception as e:
        print(f"    [Audio Error] {e}")
        return 0, duration_limit

def create_blurred_background(image_path, duration):
    """Creates a 1080x1920 blurred vertical background from cover art."""
    img = Image.open(image_path).convert("RGB")
    
    # Crop to 9:16 aspect ratio to fill screen
    target_ratio = 1080 / 1920
    img_ratio = img.width / img.height
    
    if img_ratio > target_ratio:
        # Image is too wide, crop width
        new_width = int(img.height * target_ratio)
        offset = (img.width - new_width) // 2
        img = img.crop((offset, 0, offset + new_width, img.height))
    else:
        # Image is too tall, crop height
        new_height = int(img.width / target_ratio)
        offset = (img.height - new_height) // 2
        img = img.crop((0, offset, img.width, offset + new_height))
        
    img = img.resize((1080, 1920), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=40)) # Heavy Blur
    
    # Darken it slightly so text pops
    # Note: MoviePy expects images in standard arrays
    return ImageClip(np.array(img)).fl_image(lambda pic: (pic * 0.4).astype('uint8')).set_duration(duration)

def generate_short(base_name):
    # 1. Locate Assets
    video_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_MASTER.mp4")
    audio_path = os.path.join(INPUT_FOLDER, f"{base_name}.mp3") 
    if not os.path.exists(audio_path): audio_path = os.path.join(INPUT_FOLDER, f"{base_name}.wav")
    
    # Check if Master Video exists
    if not os.path.exists(video_path):
        print(f"    [Error] Master Video not found: {video_path}")
        print(f"    Run 'main.py' first to generate the 1080p Master.")
        return

    # 2. Find Chorus
    start, end = find_chorus_window(audio_path)
    duration = end - start

    # 3. Prepare Background (Blurred Art)
    # Try to find a JPG/PNG
    potential_art = [f for f in os.listdir(INPUT_FOLDER) if base_name in f and f.lower().endswith(('.jpg', '.png'))]
    if potential_art:
        bg_clip = create_blurred_background(os.path.join(INPUT_FOLDER, potential_art[0]), duration)
    else:
        bg_clip = ColorClip(size=(1080, 1920), color=(20, 20, 20), duration=duration)

    # 4. Prepare Foreground (The Master Video)
    # We crop the center 1080x1080 square. 
    # Original is 1920x1080.
    # X-Crop: (1920-1080)/2 = 420. So we keep x=420 to x=1500.
    video = VideoFileClip(video_path).subclip(start, end)
    
    # Crop to Square (keeps lyrics and visualizer)
    video_square = video.crop(x1=420, y1=0, x2=1500, y2=1080)
    
    # Position in dead center of vertical frame
    video_square = video_square.set_position(("center", "center"))

    # 5. Add New Watermark (Bottom)
    try:
        wm = TextClip(WATERMARK, fontsize=60, color='white', font=FONT_PATH, stroke_color='black', stroke_width=2)
        wm = wm.set_position(('center', 1600)).set_duration(duration)
        layers = [bg_clip, video_square, wm]
    except Exception as e:
        print(f"    [Font Error] {e}. Using no watermark.")
        layers = [bg_clip, video_square]
    
    # 6. Render
    final = CompositeVideoClip(layers, size=(1080, 1920))
    final.set_audio(video.audio) # Use video audio to ensure sync
    
    output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_SHORT.mp4")
    final.write_videofile(output_file, fps=30, codec='libx264', audio_codec='aac', 
                          ffmpeg_params=['-pix_fmt', 'yuv420p'], threads=4)
    print(f"    [SUCCESS] Saved to {output_file}")

if __name__ == "__main__":
    # BATCH MODE: Process ALL audio files in the folder
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: '{INPUT_FOLDER}' directory not found.")
        exit()

    audio_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp3', '.wav'))]
    
    if audio_files:
        print(f"--- [Shorts Batch Started] Found {len(audio_files)} songs ---")
        for i, filename in enumerate(audio_files):
            try:
                base_name = os.path.splitext(filename)[0]
                print(f"\n[{i+1}/{len(audio_files)}] Processing: {base_name}")
                generate_short(base_name)
            except Exception as e:
                print(f"    [Skipped] Error processing {filename}: {e}")
        print(f"\n--- [Batch Complete] ---")
    else:
        print(f"No .mp3 or .wav files found in {INPUT_FOLDER}/")