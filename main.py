"""
==============================================================================
SCRIPT: VinCreationz Batch Video Generator (v13.1 - Production Master)
v13.1 FINAL: Clean border engine + Shadowed Suno Text + 4070 NVENC Optimized.
==============================================================================
"""

import librosa
import numpy as np
import PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageStat, PIL.ImageFont, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageChops
import os, time, datetime, glob, random, re, cv2
from tqdm import tqdm
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_audioclips
import moviepy.video.fx.all as vfx 
from moviepy.config import change_settings
from multiprocessing import Pool

# ==========================================
# 0. CONFIGURATION & FONT LOADING
# ==========================================
def load_config(config_path="config.txt"):
    config = {
        "TEST_MODE": True,
        "RENDER_VIDEO": True,
        "RANDOM_CROP_IMAGES": False,
        "ARTIST_NAME": "VinCreationz",
        "WATERMARK_TEXT": "VinCreationz",
        "CHINESE_FONT_PATH": "msyhbd.ttc",
        "LRC_OFFSET": 5.0
    }
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip(): continue
                key, value = line.strip().split("=", 1)
                val = value.strip().lower()
                config[key.strip()] = True if val == "true" else False if val == "false" else value.strip()
    return config

CONFIG = load_config()
CHINESE_FONT = CONFIG["CHINESE_FONT_PATH"]
WATERMARK_TEXT = CONFIG["WATERMARK_TEXT"]
ARTIST_NAME = CONFIG["ARTIST_NAME"]
TEST_MODE = CONFIG["TEST_MODE"]
RENDER_VIDEO = CONFIG["RENDER_VIDEO"]

change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})
INPUT_FOLDER, OUTPUT_FOLDER = "test_assets", "batch_renders"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 1. CORE VISUAL ENGINES (Clean Border)
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
    
    # Apply Clean 2px White Border (Replaces Shadow)
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
    
    def render_slide(img, progress):
        zoom_amt = 0.05 if CONFIG["RANDOM_CROP_IMAGES"] else 0.1 
        scale = 1.0 + (zoom_amt * progress)
        h, w = img.shape[:2]
        new_w, new_h = int(w / scale), int(h / scale)
        x, y = (w - new_w) // 2, (h - new_h) // 2
        return cv2.resize(img[y:y+new_h, x:x+new_w], (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

    current_frame = render_slide(images[slide_idx], t_local / duration_per_slide)
    time_left = duration_per_slide - t_local
    if time_left < transition_time and slide_idx < n_images - 1:
        alpha = 1.0 - (time_left / transition_time)
        current_frame = cv2.addWeighted(current_frame, 1.0 - alpha, render_slide(images[slide_idx + 1], 0.0), alpha, 0)
    return current_frame

def apply_dynamic_fx(base_frame, t, onset_env, duration, recipe):
    idx = int(t * (len(onset_env) / duration))
    pulse = onset_env[idx] if idx < len(onset_env) else 0
    if pulse < 0.05: return base_frame 
    frame = base_frame.astype(np.float32)
    if recipe["zoom"] > 0:
        z = 1.0 + (pulse * recipe["zoom"])
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w * z), int(h * z)), interpolation=cv2.INTER_LANCZOS4)
        cy, cx = (resized.shape[0]-h)//2, (resized.shape[1]-w)//2
        frame = resized[cy:cy+h, cx:cx+w]
    return np.clip(frame, 0, 255).astype(np.uint8)

# ==========================================
# 2. HELPERS & ART GENERATORS (Shadow Logic)
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
                else:
                    cleaned_lines.append(clean_line)
            elif cleaned_lines:
                cleaned_lines[-1] = f"{cleaned_lines[-1]} {clean_line}"
        with open(lrc_path, 'w', encoding='utf-8-sig') as f:
            f.write("\n".join(cleaned_lines) + "\n")
        print(f"    [LRC Sanitized] SUCCESS: {os.path.basename(lrc_path)}")
    except Exception as e: print(f"    [Cleanup Error] {e}")

def create_streaming_cover_art(source_image_path, base_name):
    try:
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_COVERART.jpg")
        img = PIL.Image.open(source_image_path).convert('RGB')
        min_side = min(img.size)
        left, top = (img.width - min_side) / 2, (img.height - min_side) / 2
        img_final = img.crop((left, top, left + min_side, top + min_side)).resize((3000, 3000), PIL.Image.LANCZOS)
        draw = PIL.ImageDraw.Draw(img_final)
        t_size = int(3000 * 0.09)
        t_font = PIL.ImageFont.truetype(CHINESE_FONT, t_size)
        raw_title = base_name.replace("_", " ").upper()
        words, wrapped_lines, cur_line = raw_title.split(), [], ""
        for word in words:
            test = f"{cur_line} {word}".strip()
            if draw.textlength(test, font=t_font) <= 2600: cur_line = test
            else:
                wrapped_lines.append(cur_line); cur_line = word
        wrapped_lines.append(cur_line)
        start_y = 1500 - ((len(wrapped_lines) * (t_size + 40)) // 2)
        for i, line in enumerate(wrapped_lines):
            w = draw.textlength(line, font=t_font)
            x, y = (3000 - w) / 2, start_y + (i * (t_size + 40))
            draw.text((x + 6, y + 6), line, font=t_font, fill="black") # Title Shadow
            draw.text((x, y), line, font=t_font, fill="white")
        wm_font = PIL.ImageFont.truetype(CHINESE_FONT, int(3000 * 0.06))
        wm_w = draw.textlength(WATERMARK_TEXT, font=wm_font)
        draw.text(((3000-wm_w)/2 + 4, 2650 + 4), WATERMARK_TEXT, font=wm_font, fill="black") # WM Shadow
        draw.text(((3000-wm_w)/2, 2650), WATERMARK_TEXT, font=wm_font, fill="white")
        img_final.save(output_path, "JPEG", quality=95)
        print(f"    [Cover Art] SUCCESS")
    except Exception as e: print(f"    [Cover Error]: {e}")

def create_suno_cover_art(source_image_path, base_name):
    try:
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_SUNO.jpg")
        img = PIL.Image.open(source_image_path).convert('RGB')
        scale = max(1080/img.width, 1920/img.height)
        new_w, new_h = int(img.width * scale + 1), int(img.height * scale + 1)
        img_resized = img.resize((new_w, new_h), PIL.Image.LANCZOS)
        left, top = (new_w - 1080) // 2, (new_h - 1920) // 2
        img_final = img_resized.crop((left, top, left + 1080, top + 1920))
        draw = PIL.ImageDraw.Draw(img_final)
        t_size, w_size = int(1080 * 0.11), int(1080 * 0.08)
        t_font, w_font = PIL.ImageFont.truetype(CHINESE_FONT, t_size), PIL.ImageFont.truetype(CHINESE_FONT, w_size)
        raw_title = base_name.replace("_", " ").upper()
        words, wrapped_lines, cur_line = raw_title.split(), [], ""
        for word in words:
            test = f"{cur_line} {word}".strip()
            if draw.textlength(test, font=t_font) <= 650: cur_line = test
            else:
                wrapped_lines.append(cur_line); cur_line = word
        wrapped_lines.append(cur_line)
        h_block = len(wrapped_lines) * (t_size + 20)
        start_y = (1920 - (h_block + 40 + w_size)) // 2
        for i, line in enumerate(wrapped_lines):
            lx = (1080 - draw.textlength(line, font=t_font)) // 2
            ly = start_y + (i * (t_size + 20))
            draw.text((lx + 4, ly + 4), line, font=t_font, fill="black") # Title Shadow
            draw.text((lx, ly), line, font=t_font, fill="white")
        wx = (1080 - draw.textlength(WATERMARK_TEXT, font=w_font)) // 2
        wy = start_y + h_block + 30
        draw.text((wx + 3, wy + 3), WATERMARK_TEXT, font=w_font, fill="black") # WM Shadow
        draw.text((wx, wy), WATERMARK_TEXT, font=w_font, fill="white")
        img_final.save(output_path, "JPEG", quality=95)
        print(f"    [Suno Art] SUCCESS")
    except Exception as e: print(f"    [Suno Error]: {e}")

def parse_lrc(lrc_path):
    lyrics = []
    if not os.path.exists(lrc_path): return []
    try:
        with open(lrc_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('['):
                    try:
                        time_tag, text = line[1:9], line[10:].strip()
                        text = re.sub(r'(issance|érique|neuroscience|decomposition)', '', text, flags=re.IGNORECASE).strip()
                        for char in ["。", "，", "─"]: text = text.replace(char, "")
                        m, s = time_tag.split(':')
                        start_time = int(m) * 60 + float(s) + float(CONFIG["LRC_OFFSET"])
                        lyrics.append({'start': start_time, 'text': text})
                    except: continue
        for i in range(len(lyrics) - 1): lyrics[i]['end'] = lyrics[i+1]['start']
        if lyrics: lyrics[-1]['end'] = lyrics[-1]['start'] + 5
    except Exception as e: print(f"LRC Error: {e}")
    return lyrics

def create_lyric_clip(lrc_master, lrc_en=None, duration=0):
    clips = []
    def contains_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    for i, line in enumerate(lrc_master):
        if not line['text'] or line['start'] > duration: continue
        is_zh = contains_chinese(line['text'])
        current_font = CHINESE_FONT if is_zh else 'Arial-Bold'
        if lrc_en and i < len(lrc_en):
            zh_txt = TextClip(line['text'], fontsize=90, color='white', font=current_font, method='caption', size=(1600, None), align='center', stroke_color='black', stroke_width=4).set_start(line['start']).set_end(line['end']).set_position(('center', 420))
            en_txt = TextClip(lrc_en[i]['text'], fontsize=50, color='yellow', font='Arial-Bold', method='caption', size=(1600, None), align='center', stroke_color='black', stroke_width=2).set_start(line['start']).set_end(line['end']).set_position(('center', 580))
            clips.extend([zh_txt, en_txt])
        else:
            txt = TextClip(line['text'], fontsize=85, color='white', font=current_font, method='caption', size=(1600, None), align='center', stroke_color='black', stroke_width=5).set_start(line['start']).set_end(line['end']).set_position(('center', 'center'))
            clips.append(txt)
    return clips

# ==========================================
# 3. CORE PROCESSOR
# ==========================================
def process_file_wrapper(args): return process_file(*args)

def process_file(audio_name, show_bar=False):
    try:
        audio_path = os.path.join(INPUT_FOLDER, audio_name)
        base_name = os.path.splitext(audio_name)[0]
        
        # Correctly synchronized LRC variable names
        lrc_master_path = os.path.join(INPUT_FOLDER, f"{base_name}.lrc")
        if os.path.exists(lrc_master_path): sanitize_and_overwrite_lrc(lrc_master_path)

        potential_files = glob.glob(os.path.join(INPUT_FOLDER, base_name + "*"))
        image_paths = sorted([f for f in potential_files if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        
        if image_paths:
            create_streaming_cover_art(image_paths[0], base_name)
            create_suno_cover_art(image_paths[0], base_name)
        
        loaded_images = []
        for p in image_paths:
            img = PIL.Image.open(p).convert('RGB')
            loaded_images.append(creative_crop_composite(img))

        duration = librosa.get_duration(path=audio_path)
        audio_limit = 20 if TEST_MODE else duration
        total_duration = audio_limit + 5 

        y, sr = librosa.load(audio_path, duration=audio_limit)
        bpm = int(np.round(librosa.beat.beat_track(y=y, sr=sr)[0]))
        recipe = {"zoom": 0.04, "expo": 0.2} if bpm >= 95 else {"zoom": 0.02, "expo": 0.1}
        c_onset = librosa.onset.onset_strength(y=y, sr=sr)
        pad_frames = int((len(c_onset)/audio_limit)*5)
        onset_env = np.concatenate([np.zeros(pad_frames), (c_onset - c_onset.min()) / (c_onset.max() - c_onset.min() + 1e-8)])

        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
        full_S_db = np.concatenate([np.zeros((80, int(S_db.shape[1]*5/audio_limit))), (S_db-S_db.min())/(S_db.max()-S_db.min()+1e-8)], axis=1)

        def make_viz(t):
            idx = min(int(t * (full_S_db.shape[1]/total_duration)), full_S_db.shape[1]-1)
            heights = np.concatenate([full_S_db[:, idx][::-1], full_S_db[:, idx]]) * 140
            img = PIL.Image.new('RGB', (1024, 180), (255, 0, 255))
            draw = PIL.ImageDraw.Draw(img)
            for i in range(160):
                if heights[i] > 2:
                    x = (1024 - (160*6))//2 + i*6
                    draw.rectangle((x, 170 - heights[i], x + 4, 170), fill=(255, int(255*(i/159)), 0))
            return np.array(img)

        bg_clip = VideoClip(lambda t: apply_dynamic_fx(get_slideshow_frame(t, loaded_images, total_duration), t, onset_env, total_duration, recipe), duration=total_duration)
        title_text = base_name.upper().replace("(", "\n(") if "(" in base_name else base_name.upper()
        title = TextClip(title_text, fontsize=110, color='white', font=CHINESE_FONT, method='caption', size=(1800, None), align='center', stroke_color='black', stroke_width=5).set_duration(5).set_position(('center', 350)).fx(vfx.fadeout, 1)
        sub = TextClip(ARTIST_NAME, fontsize=75, color='white', font=CHINESE_FONT, stroke_color='black', stroke_width=3).set_duration(5).set_position(('center', 350 + title.size[1] + 20)).fx(vfx.fadeout, 1)
        mark = TextClip(WATERMARK_TEXT, fontsize=50, color='white', font=CHINESE_FONT).set_start(5).set_duration(total_duration-5).set_position((1540, 1000)).fx(vfx.fadein, 1)
        viz_clip = VideoClip(make_viz, duration=total_duration).fx(vfx.mask_color, color=[255, 0, 255], thr=50, s=5).set_position(('center', 900))
        
        layers = [bg_clip, viz_clip, title, sub, mark]
        lrc_master = parse_lrc(lrc_master_path)
        if lrc_master: layers.extend(create_lyric_clip(lrc_master, parse_lrc(os.path.join(INPUT_FOLDER, f"{base_name}_en.lrc")), total_duration))
        
        final = CompositeVideoClip(layers, size=(1920, 1080))
        PIL.Image.fromarray(final.get_frame(2.0)).save(os.path.join(OUTPUT_FOLDER, f"{base_name}_thumb.jpg"), "JPEG", quality=90)

        if not RENDER_VIDEO: return True
        audio = concatenate_audioclips([AudioFileClip(audio_path).subclip(0, 0.1).volumex(0).set_duration(5), AudioFileClip(audio_path).subclip(0, audio_limit)])
        final.set_audio(audio).write_videofile(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_MASTER.mp4"), 
            fps=30, codec='h264_nvenc', ffmpeg_params=['-preset', 'p7', '-rc', 'vbr', '-cq', '20', '-b:v', '0', '-pix_fmt', 'yuv420p'], 
            logger='bar' if show_bar else None
        )
        return True
    except Exception as e: print(f"Error: {e}"); return False

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    audio_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp3', '.wav'))]
    if audio_files:
        start_time = time.time()
        print(f"\n[{ARTIST_NAME}] v13.1 Production Start | Clean Border & Shadows Engaged\n" + "="*40)
        if len(audio_files) == 1: process_file(audio_files[0], show_bar=True)
        else:
            with Pool(2) as pool: list(tqdm(pool.imap_unordered(process_file_wrapper, [(f, False) for f in audio_files]), total=len(audio_files)))