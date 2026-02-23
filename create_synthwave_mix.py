"""
SCRIPT: VinCreationz Synthwave Mix Generator (vFinal - Clean Output)
PURPOSE: Stitches audio with crossfades + loops video.
FIXES: Suppresses [WinError 6] at the end of the script.
"""

import os
import sys

# --- THE "OCD FIX" (Monkey Patch) ---
# This overrides the default cleanup method to ignore "Handle is invalid" errors.
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
def safe_del(self):
    try:
        self.close()
    except Exception:
        pass
FFMPEG_VideoReader.__del__ = safe_del
# ------------------------------------

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
from moviepy.config import change_settings

# Point to ImageMagick (Required for TextClip if needed later)
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})

# ================= CONFIGURATION =================
TEST_MODE = True        # Set True for transition test, False for Full Mix
CROSSFADE_SEC = 3.0     # Duration of overlap/fade
# =================================================

ASSET_FOLDER = r"test_assets\synthwave"
OUTPUT_FILENAME = "Synthwave_Mix_Vol1.mp4"
VIDEO_LOOP_NAME = "Neon_Grid_Loop.mp4"  

def create_mix():
    print(f"🎹 Starting Synthwave Mix Generator (Test Mode: {TEST_MODE})...")

    # 1. LOAD AUDIO FILES
    audio_extensions = ('.wav', '.mp3')
    audio_files = sorted([f for f in os.listdir(ASSET_FOLDER) if f.lower().endswith(audio_extensions)])
    
    if not audio_files:
        print(f"❌ No audio files found in {ASSET_FOLDER}")
        return

    # --- TEST MODE PREP ---
    if TEST_MODE:
        print("✂️  TEST MODE: Grabbing only first 2 tracks for transition check.")
        audio_files = audio_files[:2] 
    # ----------------------

    print(f"📂 Processing {len(audio_files)} tracks...")

    audio_clips = []
    current_start = 0.0

    for i, fname in enumerate(audio_files):
        path = os.path.join(ASSET_FOLDER, fname)
        clip = AudioFileClip(path)
        
        # --- TEST MODE SLICING ---
        if TEST_MODE:
            if i == 0: 
                # First song: Keep only last 10 seconds
                clip = clip.subclip(clip.duration - 10, clip.duration)
            elif i == 1:
                # Second song: Keep only first 10 seconds
                clip = clip.subclip(0, 10)
        # -------------------------

        # Apply Fades
        if i > 0:
            clip = clip.audio_fadein(CROSSFADE_SEC)
        if i < len(audio_files) - 1:
            clip = clip.audio_fadeout(CROSSFADE_SEC)
            
        # Set Start Time
        clip = clip.set_start(current_start)
        audio_clips.append(clip)
        
        current_start += clip.duration - CROSSFADE_SEC

    # 2. COMBINE AUDIO
    final_audio = CompositeAudioClip(audio_clips)
    total_duration = final_audio.duration
    
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds")

    # 3. LOAD & LOOP VIDEO
    video_path = os.path.join(ASSET_FOLDER, VIDEO_LOOP_NAME)
    if not os.path.exists(video_path):
        print(f"❌ Video loop not found: {VIDEO_LOOP_NAME}")
        return

    print("📼 Looping video visual...")
    # Load and mute video to avoid any hidden audio streams
    video_clip = VideoFileClip(video_path).without_audio()
    
    if video_clip.h != 1080:
        video_clip = video_clip.resize(height=1080)
        
    final_video = video_clip.loop(duration=total_duration)
    
    # 4. COMBINE & EXPORT
    final_video = final_video.set_audio(final_audio)
    
    output_path = os.path.join(ASSET_FOLDER, "TEST_TRANSITION.mp4" if TEST_MODE else OUTPUT_FILENAME)
    print(f"🚀 Rendering to: {output_path}")
    
    final_video.write_videofile(
        output_path, 
        fps=24, 
        codec='libx264', 
        audio_codec='aac', 
        threads=4, 
        preset='ultrafast' if TEST_MODE else 'fast',
        verbose=False,
        logger='bar' # Keeps the progress bar but hides internal ffmpeg spam
    )
    
    # Explicit close to be polite to the system
    final_video.close()
    if TEST_MODE:
        print("✅ Transition Test Complete! Check the file to hear the crossfade.")
    else:
        print("✅ Full Mix Complete!")

if __name__ == "__main__":
    create_mix()