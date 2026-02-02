"""
==============================================================================
SCRIPT: VinCreationz Batch Video Generator (v10.3 - Production)
AUTHOR: VinCreationz + Gemini
PURPOSE: Automates 1080p music videos AND 3000px DistroKid Cover Art.

------------------------------------------------------------------------------
UPDATES (v10.3):
- Integrated Text Overlay for Cover Art (Song Title + VinCreationz Watermark)
- Added PIL.ImageFont to imports
==============================================================================
"""

import librosa  # The "Ears": Analyzes audio for BPM and beat drops
import numpy as np  # The "Calculator": Handles heavy math arrays efficiently
import PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageStat, PIL.ImageFont  # The "Canvas": Image manipulation
import os  # The "Manager": Finds files and creates folders
import time
import datetime
import glob
from tqdm import tqdm  # The "Progress Bar": Shows the loading bar in terminal
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_audioclips
import moviepy.video.fx.all as vfx 
from moviepy.config import change_settings
import cv2  # The "Heavy Lifter": Advanced computer vision library
from multiprocessing import Pool  # The "Multi-tasker": Allows using multiple CPU cores

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
FPS = 30  # 30 is standard. 60 doubles render time.
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})

INPUT_FOLDER = "test_assets" 
OUTPUT_FOLDER = "batch_renders"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# TEST_MODE: True = 20s render. False = Full song.
TEST_MODE = True  

# ==========================================
# 2. MULTIPROCESSING (SPEED)
# ==========================================
USE_MULTIPROCESSING = True 
MAX_PROCESSES = 2 

# ==========================================
# 3. NEW: STREAMING COVER ART GENERATOR
# ==========================================
def create_streaming_cover_art(source_image_path, base_name):
    """
    Takes the first background image, crops it to a perfect square from the center,
    upscales it to 3000x3000px, adds Song Title & Watermark, and saves it.
    """
    try:
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_COVERART.jpg")
        img = PIL.Image.open(source_image_path).convert('RGB')
        
        # 1. Determine shortest side for a square crop
        min_side = min(img.size)
        
        # 2. Calculate center crop coordinates
        left = (img.width - min_side) / 2
        top = (img.height - min_side) / 2
        right = (img.width + min_side) / 2
        bottom = (img.height + min_side) / 2
        
        # 3. Crop to square
        img_cropped = img.crop((left, top, right, bottom))
        
        # 4. Upscale to 3000x3000 using high-quality Lanczos filter
        img_final = img_cropped.resize((3000, 3000), PIL.Image.LANCZOS)
        
        # --- TEXT OVERLAY LOGIC ---
        draw = PIL.ImageDraw.Draw(img_final)
        width, height = img_final.size
        
        # Text Configuration
        # Cleans filename "My_Song_Name" -> "My Song Name"
        song_title = base_name.replace("_", " ").replace("-", " ").title() 
        watermark_text = "VinCreationz"
        
        # Dynamic Font Sizing (Based on 3000px width)
        title_size = int(width * 0.1)   # Title is 10% of width (300px)
        wm_size = int(width * 0.04)     # Watermark is 4% of width (120px)
        
        # Font Loading
        font_name = "arial.ttf" # Windows standard
        try:
            title_font = PIL.ImageFont.truetype(font_name, title_size)
            wm_font = PIL.ImageFont.truetype(font_name, wm_size)
        except IOError:
            # Fallback if arial.ttf isn't found
            title_font = PIL.ImageFont.load_default()
            wm_font = PIL.ImageFont.load_default()
            print("   [Cover Art] Warning: Custom font not found, using default.")

        # Position Calculations using textbbox (more accurate than textsize)
        # Title Position (Centered, slightly above middle)
        try:
            bbox = draw.textbbox((0, 0), song_title, font=title_font)
            t_w = bbox[2] - bbox[0]
            t_h = bbox[3] - bbox[1]
        except AttributeError:
             # Fallback for older Pillow versions
            t_w, t_h = draw.textsize(song_title, font=title_font)

        t_x = (width - t_w) / 2
        t_y = (height - t_h) / 2 - (height * 0.05) 
        
        # Watermark Position (Centered, below title)
        try:
            wm_bbox = draw.textbbox((0, 0), watermark_text, font=wm_font)
            wm_w = wm_bbox[2] - wm_bbox[0]
            wm_h = wm_bbox[3] - wm_bbox[1]
        except AttributeError:
            wm_w, wm_h = draw.textsize(watermark_text, font=wm_font)

        wm_x = (width - wm_w) / 2
        wm_y = t_y + t_h + 60 # 60px padding below title

        # Drawing with Shadow
        shadow_offset = int(width * 0.003) # ~9px shadow
        shadow_color = (0, 0, 0)
        text_color = (255, 255, 255)
        
        # Draw Title
        draw.text((t_x + shadow_offset, t_y + shadow_offset), song_title, font=title_font, fill=shadow_color)
        draw.text((t_x, t_y), song_title, font=title_font, fill=text_color)
        
        # Draw Watermark
        draw.text((wm_x + shadow_offset, wm_y + shadow_offset), watermark_text, font=wm_font, fill=shadow_color)
        draw.text((wm_x, wm_y), watermark_text, font=wm_font, fill=text_color)
        
        # 5. Save high quality JPEG
        img_final.save(output_path, "JPEG", quality=98)
        print(f"   [Cover Art] Saved 3000px cover with text: {output_path}")
        
    except Exception as e:
        print(f"   [Cover Art Error]: {e}")

# ==========================================
# 4. THE SLIDESHOW ENGINE (The "Alive" Logic)
# ==========================================
def get_slideshow_frame(t, images, total_duration, transition_time=2.0):
    """
    Determines exactly what the background pixel data should be at time 't'.
    Handles Image Order, Ken Burns Zoom, and Crossfading.
    """
    n_images = len(images)
    if n_images == 0: return np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Timing Math
    duration_per_slide = total_duration / n_images
    slide_idx = int(t // duration_per_slide)
    slide_idx = min(slide_idx, n_images - 1)
    t_local = t % duration_per_slide
    
    # Inner Function: Draw with Zoom
    def render_slide(img, progress):
        scale = 1.0 + (0.1 * progress) # Zoom from 100% to 110%
        h, w = img.shape[:2]
        new_w, new_h = int(w / scale), int(h / scale)
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        crop = img[y:y+new_h, x:x+new_w]
        return cv2.resize(crop, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

    # Render current frame
    progress = t_local / duration_per_slide
    current_frame = render_slide(images[slide_idx], progress)
    
    # Crossfade Logic
    time_left = duration_per_slide - t_local
    if time_left < transition_time and slide_idx < n_images - 1:
        next_slide_idx = slide_idx + 1
        alpha = 1.0 - (time_left / transition_time)
        next_frame = render_slide(images[next_slide_idx], 0.0)
        current_frame = cv2.addWeighted(current_frame, 1.0 - alpha, next_frame, alpha, 0)

    return current_frame

# ==========================================
# 5. FX ENGINE (The Beat Reaction)
# ==========================================
def apply_dynamic_fx(base_frame, t, onset_env, duration, recipe):
    """
    Takes the finished background frame and adds the "Beat Pulse" on top.
    """
    idx = int(t * (len(onset_env) / duration))
    pulse = onset_env[idx] if idx < len(onset_env) else 0
    
    if pulse < 0.05: return base_frame # Skip if quiet
    
    frame = base_frame.astype(np.float32)
    
    # FX 1: Brightness Flash
    if recipe["expo"] > 0: frame *= (1.0 + pulse * recipe["expo"])
    
    # FX 2: Beat Zoom (Punch)
    if recipe["zoom"] > 0:
        z = 1.0 + (pulse * recipe["zoom"])
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w * z), int(h * z)), interpolation=cv2.INTER_LANCZOS4)
        crop_y = (resized.shape[0] - h) // 2
        crop_x = (resized.shape[1] - w) // 2
        frame = resized[crop_y:crop_y + h, crop_x:crop_x + w]
        
    return np.clip(frame, 0, 255).astype(np.uint8)

# ==========================================
# 6. HELPER TOOLS (Analysis & Parsing)
# ==========================================
def analyze_image_vibe(img_pil):
    """Decides if the image is bright/vibrant to tune FX intensity."""
    stat = PIL.ImageStat.Stat(img_pil)
    brightness = sum(stat.mean) / 3 
    extrema = img_pil.getextrema()
    color_range = sum([ex[1] - ex[0] for ex in extrema]) / 3
    is_vibrant = brightness > 127 and color_range > 100
    return brightness, is_vibrant

def get_fx_recipe(bpm, brightness, is_vibrant):
    """Sets FX intensity based on BPM."""
    recipe = {"zoom": 0.02, "expo": 0.1, "blur": 0, "rgb": 0}
    if bpm >= 125:
        recipe.update({"zoom": 0.07, "expo": 0.35, "blur": 3 if not is_vibrant else 1, "rgb": 12})
    elif bpm >= 95:
        recipe.update({"zoom": 0.04, "expo": 0.2, "blur": 0, "rgb": 0})
    if brightness > 170:
        recipe["expo"] *= 0.4
    return recipe

def parse_lrc(lrc_path):
    """Parses .LRC files into Python lists."""
    lyrics = []
    OFFSET = 5.0 
    if not os.path.exists(lrc_path): return []
    try:
        with open(lrc_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('['):
                    try:
                        time_tag = line[1:9]
                        text = line[10:].strip()
                        m, s = time_tag.split(':')
                        start = int(m) * 60 + float(s) + OFFSET
                        lyrics.append({'start': start, 'text': text})
                    except: continue
        for i in range(len(lyrics) - 1):
            lyrics[i]['end'] = lyrics[i+1]['start']
        if lyrics:
            lyrics[-1]['end'] = lyrics[-1]['start'] + 5
    except Exception as e: print(f"LRC Error: {e}")
    return lyrics

def create_lyric_clip(lyrics, duration):
    """Converts lyrics to TextClips."""
    clips = []
    for line in lyrics:
        if not line['text'] or line['start'] > duration: continue
        txt = TextClip(line['text'], fontsize=90, color='white', font='Arial-Bold', 
                       method='caption', size=(1600, None), align='center', 
                       stroke_color='black', stroke_width=4)
        txt = txt.set_start(line['start']).set_end(line['end']).set_position(('center', 'center'))
        clips.append(txt)
    return clips

# ==========================================
# 7. MAIN PROCESSING LOOP
# ==========================================
def process_file_wrapper(args): return process_file(*args)

def process_file(audio_name, show_bar=False):
    try:
        # Paths
        audio_path = os.path.join(INPUT_FOLDER, audio_name)
        base_name = os.path.splitext(audio_name)[0]
        
        # --- STEP A: FIND MATCHING IMAGES ---
        search_pattern = os.path.join(INPUT_FOLDER, base_name + "*")
        potential_files = glob.glob(search_pattern)
        valid_exts = ['.jpg', '.png', '.jpeg', '.webp']
        image_paths = [f for f in potential_files if os.path.splitext(f)[1].lower() in valid_exts]
        image_paths.sort() 
        
        if not image_paths: 
            print(f"Skipping {base_name}: No images found.")
            return False

        # --- STEP B: GENERATE DISTROKID COVER ART ---
        # (New Feature: Creates 3000px square cover with text)
        create_streaming_cover_art(image_paths[0], base_name)

        # --- STEP C: PRELOAD VIDEO IMAGES ---
        loaded_images = []
        for p in image_paths:
            img = PIL.Image.open(p).convert('RGB')
            orig_w, orig_h = img.size
            scale = max(1920 / orig_w, 1080 / orig_h)
            new_w, new_h = int(orig_w * scale + 1), int(orig_h * scale + 1)
            img = img.resize((new_w, new_h), PIL.Image.LANCZOS)
            left = (new_w - 1920)/2
            top = (new_h - 1080)/2
            img = img.crop((left, top, left+1920, top+1080))
            loaded_images.append(np.array(img))

        # --- STEP D: AUDIO ANALYSIS ---
        duration = librosa.get_duration(path=audio_path)
        audio_limit = 15 if TEST_MODE else duration
        total_duration = audio_limit + 5 

        y, sr = librosa.load(audio_path, duration=audio_limit)
        bpm = int(np.round(librosa.beat.beat_track(y=y, sr=sr)[0]))
        
        bright, vibr = analyze_image_vibe(PIL.Image.fromarray(loaded_images[0]))
        recipe = get_fx_recipe(bpm, bright, vibr)

        c_onset = librosa.onset.onset_strength(y=y, sr=sr)
        pad_frames = int((len(c_onset) / audio_limit) * 5)
        onset_env = np.concatenate([np.zeros(pad_frames), c_onset])
        onset_env = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-8)

        # Spectrogram for Visualizer
        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
        pad_spec = int(S_db.shape[1] * 5 / audio_limit)
        full_S_db = np.concatenate([np.zeros((80, pad_spec)), (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)], axis=1)

        # --- STEP E: DRAW VISUALIZER ---
        viz_w, viz_h = 1024, 180  
        bar_count = 160
        colors = [(255, int(255 * (i / (bar_count - 1))), 0) for i in range(bar_count)]
        
        def make_viz(t):
            idx = int(t * (full_S_db.shape[1] / total_duration))
            curr = full_S_db[:, min(idx, full_S_db.shape[1] - 1)]
            mirr = np.concatenate([curr[::-1], curr])
            heights = mirr * (viz_h - 40)
            
            img = PIL.Image.new('RGB', (viz_w, viz_h), (255, 0, 255))
            draw = PIL.ImageDraw.Draw(img)
            for i in range(bar_count):
                h = heights[i]
                if h <= 2: continue
                x = (viz_w - (bar_count*6)) // 2 + i * 6
                draw.rectangle((x, viz_h - 10 - h, x + 4, viz_h - 10), fill=colors[i])
            return np.array(img)

        # --- STEP F: VIDEO CONSTRUCTION ---
        def video_generator(t):
            base = get_slideshow_frame(t, loaded_images, total_duration)
            final = apply_dynamic_fx(base, t, onset_env, total_duration, recipe)
            return final

        bg_clip = VideoClip(video_generator, duration=total_duration)
        
        # Text Overlays
        title = TextClip(base_name.upper(), fontsize=110, color='white', font='Arial-Bold', 
                         method='caption', size=(1800, None), 
                         stroke_color='black', stroke_width=5) \
                .set_duration(5).set_position(('center', 400)).fx(vfx.fadeout, 1)

        sub = TextClip("VinCreationz", fontsize=50, color='white', font='Arial-Bold',
                       stroke_color='black', stroke_width=2) \
              .set_duration(5).set_position(('center', 620)).fx(vfx.fadeout, 1)

        # Smart Watermark (At X=1540)
        wm_region = loaded_images[0][1000:1050, 1540:1850] 
        wm_brightness = np.mean(wm_region)
        wm_color = 'black' if wm_brightness > 140 else 'white'
        
        mark = TextClip("VinCreationz", fontsize=50, color=wm_color, font='Arial-Bold') \
               .set_start(5).set_duration(total_duration - 5) \
               .set_position((1540, 1000)).fx(vfx.fadein, 1)

        viz_clip = VideoClip(make_viz, duration=total_duration).fx(vfx.mask_color, color=[255, 0, 255], thr=50, s=5)
        
        layers = [bg_clip, viz_clip.set_position(('center', 900)), title, sub, mark]
        
        # Add Lyrics
        lrc_path = os.path.join(INPUT_FOLDER, base_name + ".lrc")
        lyric_clips = parse_lrc(lrc_path)
        if lyric_clips:
            text_layers = create_lyric_clip(lyric_clips, total_duration)
            layers.extend(text_layers)
        
        # --- STEP G: RENDER ---
        final = CompositeVideoClip(layers, size=(1920, 1080))
        audio = concatenate_audioclips([AudioFileClip(audio_path).subclip(0, 0.1).volumex(0).set_duration(5), AudioFileClip(audio_path).subclip(0, audio_limit)])

        # Auto-Thumbnail
        thumb_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_thumb.jpg")
        try:
            frame_data = final.get_frame(2.0)
            thumb_img = PIL.Image.fromarray(frame_data)
            thumb_img.save(thumb_path, "JPEG", quality=95)
            if os.path.getsize(thumb_path) > 2 * 1024 * 1024:
                thumb_img.save(thumb_path, "JPEG", quality=85)
        except Exception as e:
            print(f"Thumbnail Error: {e}")

        # Render Video
        final.set_audio(audio).write_videofile(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_MASTER.mp4"), 
            fps=FPS, codec='h264_nvenc', threads=os.cpu_count(), 
            ffmpeg_params=['-preset', 'p7', '-b:v', '20M'], 
            logger='bar' if show_bar else None
        )
        return True
    except Exception as e: print(f"Error: {e}"); return False

# ==========================================
# 8. SCRIPT EXECUTION START
# ==========================================
if __name__ == "__main__":
    # Now accepts both MP3 and WAV
    audio_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp3', '.wav'))]
    total = len(audio_files)

    if total == 0:
        print("No MP3 files found in input folder!")
    else:
        start_time = time.time()
        start_str = datetime.datetime.now().strftime("%I:%M:%S %p")
        print(f"\n[VinCreationz] Batch Started at: {start_str}")
        print(f"Queue Size: {total} files")
        print("="*40)

        if total == 1:
            process_file(audio_files[0], show_bar=True)
        else:
            if USE_MULTIPROCESSING:
                with Pool(min(MAX_PROCESSES, total)) as pool:
                    list(tqdm(pool.imap_unordered(process_file_wrapper, [(f, False) for f in audio_files]), total=total))
            else:
                for f in audio_files: process_file(f, show_bar=True)

        end_time = time.time()
        duration = end_time - start_time
        print("\n" + "="*40)
        print("      VINCREATIONZ PRODUCTION REPORT      ")
        print(f"Total Duration: {str(datetime.timedelta(seconds=int(duration)))}")
        print("="*40 + "\n")