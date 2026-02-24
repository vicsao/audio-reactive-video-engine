"""
==============================================================================
SCRIPT: VinCreationz Batch Video Generator (v13.2.3 - SolutionLand Master)
v13.2.3: Config Sync + Rainbow Viz + Performance Telemetry + Random Pan/Zoom.
==============================================================================
"""

import librosa
import numpy as np
import PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageStat, PIL.ImageFont, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageChops
import os, time, datetime, glob, random, re, cv2, shutil
from tqdm import tqdm
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_audioclips
import moviepy.video.fx.all as vfx 
from moviepy.config import change_settings
from multiprocessing import Pool

# ==========================================
# 0. CONFIGURATION SYNC
# ==========================================
def load_config(config_path="config.txt"):
    config = {
        "TEST_MODE": True,
        "RENDER_VIDEO": True,
        "RANDOM_CROP_IMAGES": True,
        "ARTIST_NAME": "VinCreationz",
        "WATERMARK_TEXT": "VinCreationz",
        "LANGUAGE_FONT_PATH": "msyhbd.ttc",
        "LRC_OFFSET": 5.0
    }
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip(): continue
                if "=" not in line: continue
                key, value = line.strip().split("=", 1)
                val = value.strip().lower()
                # Handle comments and defaults in the config value
                clean_val = value.split('(')[0].strip() if '(' in value else value.strip()
                config[key.strip()] = True if val == "true" else False if val == "false" else clean_val
    return config

CONFIG = load_config()
LANGUAGE_FONT = CONFIG["LANGUAGE_FONT_PATH"]
WATERMARK_TEXT = CONFIG["WATERMARK_TEXT"]
ARTIST_NAME = CONFIG["ARTIST_NAME"]
TEST_MODE = CONFIG["TEST_MODE"]
RENDER_VIDEO = CONFIG["RENDER_VIDEO"]

change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})
INPUT_FOLDER, OUTPUT_FOLDER = "test_assets", "batch_renders"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 1. HELPERS & SANITIZATION
# ==========================================
def sanitize_and_overwrite_lrc(lrc_path):
    if not os.path.exists(lrc_path): return
    cleaned_lines = []
    timestamp_pattern = re.compile(r"\[\d{2}:\d{2}\.\d{2}\]") 
    try:
        with open(lrc_path, 'r', encoding='utf-8-sig') as f:
            raw_lines = f.readlines()
        for line in raw_lines:
            clean_line = line.strip()
            if not clean_line: continue
            text_only = re.sub(r"\[\d{2}:\d{2}\.\d{2}\]", "", clean_line).strip()
            if timestamp_pattern.search(clean_line):
                if cleaned_lines and (len(text_only) < 12 or text_only.startswith('(')):
                    cleaned_lines[-1] = f"{cleaned_lines[-1]} {text_only}"
                else: cleaned_lines.append(clean_line)
            elif cleaned_lines: cleaned_lines[-1] = f"{cleaned_lines[-1]} {clean_line}"
        with open(lrc_path, 'w', encoding='utf-8-sig') as f:
            f.write("\n".join(cleaned_lines) + "\n")
        print(f"    [LRC Sanitized] SUCCESS: {os.path.basename(lrc_path)}")
    except Exception as e: print(f"    [Cleanup Error] {e}")

# ==========================================
# 2. CORE VISUAL ENGINES
# ==========================================
def creative_crop_composite(img_pil):
    bg_w, bg_h = img_pil.size
    scale = max(1920 / bg_w, 1080 / bg_h)
    bg_img = img_pil.resize((int(bg_w * scale + 1), int(bg_h * scale + 1)), PIL.Image.LANCZOS)
    left, top = (bg_img.width - 1920) / 2, (bg_img.height - 1080) / 2
    bg_img = bg_img.crop((left, top, left + 1920, top + 1080))
    bg_img = bg_img.filter(PIL.ImageFilter.GaussianBlur(radius=30))
    bg_img = PIL.ImageEnhance.Brightness(bg_img).enhance(0.5)

    mode = random.randint(1, 3) 
    target_size = (920, 920) if mode == 1 else (1220, 820) if mode == 2 else (620, 970)
    t_w, t_h = target_size
    orig_w, orig_h = img_pil.size
    target_aspect, img_aspect = t_w / t_h, orig_w / orig_h

    if img_aspect > target_aspect:
        new_h, new_w = orig_h, int(orig_h * target_aspect)
    else:
        new_w, new_h = orig_w, int(orig_w / target_aspect)

    left, top = (orig_w - new_w) / 2, (orig_h - new_h) / 2
    fg_img = img_pil.crop((left, top, left + new_w, top + new_h)).resize(target_size, PIL.Image.LANCZOS)
    fg_with_border = PIL.ImageOps.expand(fg_img, border=2, fill='white')
    pos_x, pos_y = (1920 - fg_with_border.width) // 2, (1080 - fg_with_border.height) // 2
    bg_img.paste(fg_with_border, (pos_x, pos_y))
    return np.array(bg_img)

def get_slideshow_frame(t, images, total_duration, transition_time=2.0):
    n_images = len(images)
    if n_images == 0: return np.zeros((1080, 1920, 3), dtype=np.uint8)
    duration_per_slide = total_duration / n_images
    slide_idx = min(int(t // duration_per_slide), n_images - 1)
    t_local = t % duration_per_slide
    
    random.seed(slide_idx) 
    pan_mode = random.randint(0, 4) if CONFIG["RANDOM_CROP_IMAGES"] else 0

    def render_slide(img, progress, mode):
        zoom_amt = 0.15 
        scale = 1.0 + (zoom_amt * progress)
        h, w = img.shape[:2]
        new_w, new_h = int(w / scale), int(h / scale)
        if mode == 1: x, y = 0, 0
        elif mode == 2: x, y = (w - new_w), 0
        elif mode == 3: x, y = 0, (h - new_h)
        elif mode == 4: x, y = (w - new_w), (h - new_h)
        else: x, y = (w - new_w) // 2, (h - new_h) // 2
        return cv2.resize(img[y:y+new_h, x:x+new_w], (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

    current_frame = render_slide(images[slide_idx], t_local / duration_per_slide, pan_mode)
    time_left = duration_per_slide - t_local
    if time_left < transition_time and slide_idx < n_images - 1:
        alpha = 1.0 - (time_left / transition_time)
        random.seed(slide_idx + 1)
        next_mode = random.randint(0, 4) if CONFIG["RANDOM_CROP_IMAGES"] else 0
        current_frame = cv2.addWeighted(current_frame, 1.0 - alpha, render_slide(images[slide_idx + 1], 0.0, next_mode), alpha, 0)
    return current_frame

def apply_dynamic_fx(base_frame, t, onset_env, duration, recipe):
    idx = int(t * (len(onset_env) / duration))
    pulse = onset_env[idx] if idx < len(onset_env) else 0
    if pulse < 0.05: return base_frame 
    frame = base_frame.astype(np.float32)
    z = 1.0 + (pulse * recipe["zoom"])
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (int(w * z), int(h * z)), interpolation=cv2.INTER_LANCZOS4)
    cy, cx = (resized.shape[0]-h)//2, (resized.shape[1]-w)//2
    return np.clip(resized[cy:cy+h, cx:cx+w], 0, 255).astype(np.uint8)

# ==========================================
# 3. ART ENGINES
# ==========================================
def create_streaming_cover_art(source_image_path, base_name):
    try:
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_COVERART.jpg")
        img = PIL.Image.open(source_image_path).convert('RGB')
        side = min(img.size)
        img_final = img.crop(((img.width-side)/2, (img.height-side)/2, (img.width+side)/2, (img.height+side)/2)).resize((3000, 3000), PIL.Image.LANCZOS)
        draw = PIL.ImageDraw.Draw(img_final)
        t_font = PIL.ImageFont.truetype(LANGUAGE_FONT, int(3000 * 0.09))
        raw_title = base_name.replace("_", " ").upper()
        words, lines, cur = raw_title.split(), [], ""
        for w in words:
            if draw.textlength(f"{cur} {w}".strip(), font=t_font) <= 2600: cur = f"{cur} {w}".strip()
            else: lines.append(cur); cur = w
        lines.append(cur)
        sy = 1500 - ((len(lines) * 270) // 2)
        for i, line in enumerate(lines):
            lx = (3000 - draw.textlength(line, font=t_font)) / 2
            draw.text((lx+6, sy+(i*270)+6), line, font=t_font, fill="black")
            draw.text((lx, sy+(i*270)), line, font=t_font, fill="white")
        img_final.save(output_path, "JPEG", quality=95)
        print(f"    [Cover Art] SUCCESS")
    except Exception as e: print(f"    [Cover Error]: {e}")

def create_suno_cover_art(source_image_path, base_name):
    try:
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_SUNO.jpg")
        img = PIL.Image.open(source_image_path).convert('RGB')
        scale = max(1080/img.width, 1920/img.height)
        img_f = img.resize((int(img.width*scale+1), int(img.height*scale+1)), PIL.Image.LANCZOS).crop(((int(img.width*scale+1)-1080)//2, (int(img.height*scale+1)-1920)//2, (int(img.width*scale+1)+1080)//2, (int(img.height*scale+1)+1920)//2))
        draw = PIL.ImageDraw.Draw(img_f)
        t_font = PIL.ImageFont.truetype(LANGUAGE_FONT, 118)
        w_font = PIL.ImageFont.truetype(LANGUAGE_FONT, 86)
        raw_title = base_name.replace("_", " ").upper()
        words, lines, cur = raw_title.split(), [], ""
        for w in words:
            if draw.textlength(f"{cur} {w}".strip(), font=t_font) <= 650: cur = f"{cur} {w}".strip()
            else: lines.append(cur); cur = w
        lines.append(cur)
        hb = len(lines) * 138
        sy = (1920 - (hb + 126)) // 2
        for i, line in enumerate(lines):
            lx = (1080 - draw.textlength(line, font=t_font)) // 2
            draw.text((lx+4, sy+(i*138)+4), line, font=t_font, fill="black")
            draw.text((lx, sy+(i*138)), line, font=t_font, fill="white")
        wx = (1080 - draw.textlength(WATERMARK_TEXT, font=w_font)) // 2
        draw.text((wx+3, sy+hb+33), WATERMARK_TEXT, font=w_font, fill="black")
        draw.text((wx, sy+hb+30), WATERMARK_TEXT, font=w_font, fill="white")
        img_f.save(output_path, "JPEG", quality=95)
        print(f"    [Suno Art] SUCCESS")
    except Exception as e: print(f"    [Suno Error]: {e}")

# ==========================================
# 4. LYRIC & PROCESSOR
# ==========================================
def parse_lrc(lrc_path):
    lyrics = []
    if not os.path.exists(lrc_path): return []
    try:
        with open(lrc_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('['):
                    try:
                        time_tag, text = line[1:9], line[10:].strip()
                        m, s = time_tag.split(':')
                        lyrics.append({'start': int(m)*60+float(s)+float(CONFIG["LRC_OFFSET"]), 'text': text})
                    except: continue
        for i in range(len(lyrics)-1): lyrics[i]['end'] = lyrics[i+1]['start']
        if lyrics: lyrics[-1]['end'] = lyrics[-1]['start']+5
    except: pass
    return lyrics

def create_lyric_clip(lrc_master, lrc_en=None, duration=0):
    clips = []
    def has_zh(text): return any('\u4e00' <= char <= '\u9fff' for char in text)
    for i, line in enumerate(lrc_master):
        if not line['text'] or line['start'] > duration: continue
        f = LANGUAGE_FONT if has_zh(line['text']) else 'Arial-Bold'
        if lrc_en and i < len(lrc_en):
            zh = TextClip(line['text'], fontsize=90, color='white', font=f, method='caption', size=(1600, None), align='center', stroke_color='black', stroke_width=4).set_start(line['start']).set_end(line['end']).set_position(('center', 420))
            en = TextClip(lrc_en[i]['text'], fontsize=50, color='yellow', font='Arial-Bold', method='caption', size=(1600, None), align='center', stroke_color='black', stroke_width=2).set_start(line['start']).set_end(line['end']).set_position(('center', 580))
            clips.extend([zh, en])
        else:
            clips.append(TextClip(line['text'], fontsize=85, color='white', font=f, method='caption', size=(1600, None), align='center', stroke_color='black', stroke_width=5).set_start(line['start']).set_end(line['end']).set_position(('center', 'center')))
    return clips

def process_file_wrapper(args): return process_file(*args)

def process_file(audio_name, show_bar=False):
    try:
        audio_path = os.path.join(INPUT_FOLDER, audio_name)
        base_name = os.path.splitext(audio_name)[0]
        lrc_master_path = os.path.join(INPUT_FOLDER, f"{base_name}.lrc")
        if os.path.exists(lrc_master_path): sanitize_and_overwrite_lrc(lrc_master_path)
        img_p = sorted([f for f in glob.glob(os.path.join(INPUT_FOLDER, base_name+"*")) if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg','.webp']])
        if img_p:
            create_streaming_cover_art(img_p[0], base_name)
            create_suno_cover_art(img_p[0], base_name)
        loaded_images = [creative_crop_composite(PIL.Image.open(p).convert('RGB')) for p in img_p]
        duration = librosa.get_duration(path=audio_path)
        audio_limit = 20 if TEST_MODE else duration
        total_duration = audio_limit + 5 

        y, sr = librosa.load(audio_path, duration=audio_limit)
        bpm = int(np.round(librosa.beat.beat_track(y=y, sr=sr)[0]))
        onset_env_raw = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env = np.concatenate([np.zeros(int(5*sr/512)), (onset_env_raw-onset_env_raw.min())/(onset_env_raw.max()-onset_env_raw.min()+1e-8)])
        
        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
        full_S_db = np.concatenate([np.zeros((80, int(S_db.shape[1]*5/audio_limit))), (S_db-S_db.min())/(S_db.max()-S_db.min()+1e-8)], axis=1)

        def make_viz(t):
            idx = min(int(t*(full_S_db.shape[1]/total_duration)), full_S_db.shape[1]-1)
            h = np.concatenate([full_S_db[:, idx][::-1], full_S_db[:, idx]]) * 140
            img = PIL.Image.new('RGB', (1024, 180), (255, 0, 255))
            draw = PIL.ImageDraw.Draw(img)
            for i in range(160):
                if h[i]>2: draw.rectangle(((1024-960)//2+i*6, 170-h[i], (1024-960)//2+i*6+4, 170), fill=(255, int(255*(i/159)), 0))
            return np.array(img)

        bg = VideoClip(lambda t: apply_dynamic_fx(get_slideshow_frame(t, loaded_images, total_duration), t, onset_env, total_duration, {"zoom":0.04 if bpm>=95 else 0.02}), duration=total_duration)
        title = TextClip(base_name.upper().replace("(","\n("), fontsize=110, color='white', font=LANGUAGE_FONT, method='caption', size=(1800, None), align='center', stroke_color='black', stroke_width=5).set_duration(5).set_position(('center', 350)).fx(vfx.fadeout, 1)
        sub = TextClip(ARTIST_NAME, fontsize=75, color='white', font=LANGUAGE_FONT, stroke_color='black', stroke_width=3).set_duration(5).set_position(('center', 350+title.size[1]+20)).fx(vfx.fadeout, 1)
        mark = TextClip(WATERMARK_TEXT, fontsize=50, color='white', font=LANGUAGE_FONT).set_start(5).set_duration(total_duration-5).set_position((1540, 1000)).fx(vfx.fadein, 1)
        viz = VideoClip(make_viz, duration=total_duration).fx(vfx.mask_color, color=[255, 0, 255], thr=50, s=5).set_position(('center', 900))
        
        layers = [bg, viz, title, sub, mark]
        lrc = parse_lrc(lrc_master_path)
        if lrc: layers.extend(create_lyric_clip(lrc, parse_lrc(os.path.join(INPUT_FOLDER, f"{base_name}_en.lrc")), total_duration))
        
        final = CompositeVideoClip(layers, size=(1920, 1080))
        PIL.Image.fromarray(final.get_frame(2.0)).save(os.path.join(OUTPUT_FOLDER, f"{base_name}_thumb.jpg"), "JPEG", quality=90)

        if RENDER_VIDEO:
            audio = concatenate_audioclips([AudioFileClip(audio_path).subclip(0, 0.1).volumex(0).set_duration(5), AudioFileClip(audio_path).subclip(0, audio_limit)])
            render_start = time.time()
            final.set_audio(audio).write_videofile(os.path.join(OUTPUT_FOLDER, f"{base_name}_MASTER.mp4"), fps=30, codec='h264_nvenc', ffmpeg_params=['-preset', 'p7', '-rc', 'vbr', '-cq', '20', '-b:v', '0', '-pix_fmt', 'yuv420p'], logger='bar' if show_bar else None)
            print(f"    [MP4 Render] SUCCESS | Time: {time.time() - render_start:.2f}s")
        return True
    except Exception as e: print(f"Error: {e}"); return False

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    audio_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp3', '.wav'))]
    if audio_files:
        script_start = time.time()
        start_clock = datetime.datetime.now().strftime("%I:%M:%S %p")
        print(f"\n[{ARTIST_NAME}] v13.2.3 SolutionLand Master | Production Suite")
        print("="*40)
        print(f" STARTED AT: {start_clock}")
        print("="*40 + "\n")
        
        if len(audio_files) == 1: process_file(audio_files[0], show_bar=True)
        else:
            with Pool(2) as pool: list(tqdm(pool.imap_unordered(process_file_wrapper, [(f, False) for f in audio_files]), total=len(audio_files)))
        
        script_end = time.time()
        end_clock = datetime.datetime.now().strftime("%I:%M:%S %p")
        total_seconds = script_end - script_start
        total_mins_content = (len(audio_files) * (20 if TEST_MODE else 240)) / 60
        approx_it_s = (len(audio_files) * (20 if TEST_MODE else 240) * 30) / total_seconds
        
        print("\n" + "="*40)
        print(f" FINAL PERFORMANCE SUMMARY")
        print(f" {'-'*25}")
        print(f" Start/End:          {start_clock} -> {end_clock}")
        print(f" Content Produced:   {total_mins_content:.2f} Minutes")
        print(f" Total Script Time:  {total_seconds:.2f}s")
        print(f" Throughput Gauge:   ~{approx_it_s:.2f} it/s")
        print("="*40 + "\n")