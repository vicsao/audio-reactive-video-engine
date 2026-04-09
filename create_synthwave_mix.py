"""
SCRIPT: SynthCreationz Mix Generator (v6.1.9 - Master Integration)
PURPOSE: Grouped config support, NVENC optimization, Source Audio Merging, and Detailed Status Summary.
FEATURES: Bloom, CA, Saturation, Film Grain, Motion Blur, Impact Engine, and UI Spectrogram.
"""

import os
import sys
import subprocess
import configparser
import random
import librosa
import shutil
import time
import io
import numpy as np 
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageChops

# --- UNICODE TERMINAL FIX ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- MOVIEPY FIXES (Preventing Handle Leaks) ---
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.audio.io.readers import FFMPEG_AudioReader

def safe_del(self):
    try:
        if hasattr(self, 'close'): self.close()
        if hasattr(self, 'proc') and self.proc: self.proc.terminate()
    except Exception: pass

FFMPEG_VideoReader.__del__ = safe_del
FFMPEG_AudioReader.__del__ = safe_del
subprocess.Popen.__del__ = lambda self: None 

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, VideoClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from moviepy.config import change_settings

# --- LOAD GROUPED CONFIG ---
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
if not os.path.exists(config_path):
    print(f"❌ ERROR: config.txt not found at {config_path}")
    sys.exit(1)

config.read(config_path)
ROOT = os.path.dirname(os.path.abspath(__file__))

# [PATHS]
RAW_ASSET = config.get('PATHS', 'asset_folder', fallback='assets').strip('\\/')
ASSET_FOLDER = os.path.join(ROOT, RAW_ASSET)
IMAGEMAGICK = config.get('PATHS', 'imagemagick_path', fallback=r'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe')

# [MASTER_SETTINGS]
TEST_MODE = config.getboolean('MASTER_SETTINGS', 'test_mode', fallback=False)
TEST_DUR = config.getint('MASTER_SETTINGS', 'test_duration', fallback=60)
FPS = config.getint('MASTER_SETTINGS', 'render_fps', fallback=30)
TARGET_H = config.getint('MASTER_SETTINGS', 'dest_vid_size', fallback=1080)
X_FADE = config.getfloat('MASTER_SETTINGS', 'crossfade_sec', fallback=3.0)

# [CINEMATIC_MASTER]
BLOOM = config.getfloat('CINEMATIC_MASTER', 'bloom_intensity', fallback=0.0)
CA_AMT = config.getfloat('CINEMATIC_MASTER', 'chromatic_aberration', fallback=0.0)
SAT = config.getfloat('CINEMATIC_MASTER', 'saturation', fallback=1.0)
GRAIN = config.getfloat('CINEMATIC_MASTER', 'film_grain', fallback=0.0)
M_BLUR = config.getfloat('CINEMATIC_MASTER', 'motion_blur', fallback=0.0)

# [IMPACT_ENGINE]
BEATS_PER_CUT = config.getint('IMPACT_ENGINE', 'beats_per_cut', fallback=16)
ENABLE_IMPACT = config.getboolean('IMPACT_ENGINE', 'enable_impact', fallback=False)
ZOOM_AMT      = config.getfloat('IMPACT_ENGINE', 'zoom_intensity', fallback=0.0)
SHAKE_AMT     = config.getint('IMPACT_ENGINE', 'shake_intensity', fallback=0)
IMPACT_DUR    = config.getfloat('IMPACT_ENGINE', 'impact_duration', fallback=0.0)

# [AUDIO_MIXER]
INC_SRC_AUDIO = config.getboolean('AUDIO_MIXER', 'include_source_audio', fallback=False)
SRC_VOL = config.getfloat('AUDIO_MIXER', 'source_audio_volume', fallback=0.3)

# [BRANDING]
ARTIST_NAME = config.get('BRANDING', 'artist_name', fallback='VinCreationz')
TRACK_PREFIX = config.get('BRANDING', 'track_label_prefix', fallback='Track:')

# GLOBAL UI & CACHE
TARGET_W = int(TARGET_H * (16/9))
UI_X_OFFSET = (TARGET_W - 580 - 50)
UI_Y_OFFSET = int(TARGET_H * 0.72)
prev_frame_pil = None 

change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK})

def analyze_rhythm(audio_path, beats_per_cut=16):
    print(f"🎧 [PHASE 1.5] Analyzing Rhythm & Energy: {os.path.basename(audio_path)}...")
    y, sr = librosa.load(audio_path, sr=44100)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(range(len(rms)), sr=44100)
    avg_rms = np.mean(rms)
    
    planned_cuts, last_cut_t = [0.0], 0.0 
    for i, t in enumerate(beat_times):
        if i % beats_per_cut == 0 and t > 0.5:
            planned_cuts.append(t)
            last_cut_t = t
            
    for i in range(10, len(rms)-10):
        t = rms_times[i]
        if rms[i] > (avg_rms * 1.6) and (t - last_cut_t > 2.0):
            planned_cuts.append(float(t))
            last_cut_t = t
            print(f"   🔥 Significant Energy Shift detected at {t:.2f}s")
    
    print(f"✅ Rhythm Analysis Complete. Found {len(planned_cuts)} sync points.")
    return sorted(list(set(planned_cuts)))

def create_mix():
    total_start = time.time()
    
    # --- DYNAMIC UI HEADERS ---
    fx_status = "ENABLED" if any([BLOOM > 0, CA_AMT > 0, SAT != 1.0, GRAIN > 0, M_BLUR > 0]) else "DISABLED"
    retro_status = "ENABLED" if config.getboolean('RETRO_FX', 'vcr_overlay', fallback=False) else "DISABLED"

    print("\n" + "="*65)
    print(f"🎹 {ARTIST_NAME.upper()} - CINEMATIC MIX ENGINE v6.1.9")
    print("="*65)
    print(f"  > [MASTER]     MODE: {'TEST' if TEST_MODE else 'FINAL'} | {TARGET_W}x{TARGET_H} @ {FPS}fps")
    print(f"  > [PATHS]      ASSETS: {RAW_ASSET}")
    print(f"  > [AUDIO]      X-FADE: {X_FADE}s | SRC-AUDIO: {INC_SRC_AUDIO} (Vol: {SRC_VOL})")
    print(f"  > [RHYTHM]     CUTS: Every {BEATS_PER_CUT} beats")
    print("-" * 65)
    print(f"  > [CINEMA]     Status: {fx_status}")
    print(f"                 Bloom: {BLOOM} | CA: {CA_AMT} | Sat: {SAT}")
    print(f"                 Grain: {GRAIN} | M-Blur: {M_BLUR}")
    print("-" * 65)
    print(f"  > [IMPACT]     Status: {'ENABLED' if ENABLE_IMPACT else 'DISABLED'}")
    print(f"                 Shake: {SHAKE_AMT} | Zoom: {ZOOM_AMT} | Dur: {IMPACT_DUR}s")
    print("-" * 65)
    print(f"  > [RETRO]      Status: {retro_status}")
    print(f"                 VCR: {config.get('RETRO_FX', 'vcr_overlay')} | Scanlines: {config.get('RETRO_FX', 'apply_scanlines')}")
    print(f"                 Glitch: {config.get('RETRO_FX', 'apply_glitch')} | V-Hold: {config.get('RETRO_FX', 'apply_vhold')}")
    print("="*65 + "\n")

    global cut_timestamps, prev_frame_pil

    # --- PHASE 1: AUDIO ASSEMBLY ---
    s1 = time.time()
    all_audio_files = sorted([f for f in os.listdir(ASSET_FOLDER) if f.lower().endswith(('.wav', '.mp3'))])
    
    if not all_audio_files:
        print("❌ CRITICAL: No audio files found!")
        return

    audio_files = all_audio_files[:2] if TEST_MODE else all_audio_files
    print(f"🎼 [PHASE 1] Merging {len(audio_files)} Tracks...")
    audio_clips, curr_t, song_boundaries = [], 0, []

    for i, fname in enumerate(audio_files):
        path = os.path.join(ASSET_FOLDER, fname)
        clip = AudioFileClip(path)
        if TEST_MODE:
            clip = clip.subclip(0, min(clip.duration, TEST_DUR / 2))
        
        if i > 0: clip = clip.audio_fadein(X_FADE)
        if i < len(audio_files) - 1: clip = clip.audio_fadeout(X_FADE)

        clip = clip.set_start(curr_t)
        audio_clips.append(clip)
        
        overlap_adj = (X_FADE - 0.1) if i < len(audio_files) - 1 else 0
        next_t = curr_t + clip.duration - overlap_adj
        song_boundaries.append((curr_t, next_t + overlap_adj, fname))
        curr_t = next_t

    final_audio = CompositeAudioClip(audio_clips)
    temp_wav = os.path.join(ASSET_FOLDER, "temp_mix_analysis.wav")
    final_audio.write_audiofile(temp_wav, fps=22050, logger=None)
    
    cut_timestamps = analyze_rhythm(temp_wav, beats_per_cut=BEATS_PER_CUT)
    t1 = time.time() - s1

    # --- PHASE 2: VIDEO CONFORMING ---
    s2 = time.time()
    src_video_dir = os.path.join(ASSET_FOLDER, "src_videos")
    if not os.path.exists(src_video_dir): os.makedirs(src_video_dir)
    
    v_files = sorted([os.path.join(src_video_dir, f) for f in os.listdir(src_video_dir) if f.lower().endswith(('.mp4', '.mov'))])
    temp_playlist_dir = os.path.join(ASSET_FOLDER, "temp_processed_playlist")
    os.makedirs(temp_playlist_dir, exist_ok=True)
    
    processed_v_paths = []
    print(f"🚀 [PHASE 2] Conforming {len(v_files)} Source Videos...")
    for i, v_path in enumerate(v_files):
        out_v = os.path.join(temp_playlist_dir, f"vid_{i:03d}.mp4")
        if not os.path.exists(out_v) or os.path.getsize(out_v) < 1000:
            print(f"  ⚙️ Processing: {os.path.basename(v_path)}...")
            ff_cmd = (f'ffmpeg -y -i "{v_path.replace("\\", "/")}" -vf '
                      f'"scale=w={TARGET_W}:h={TARGET_H}:force_original_aspect_ratio=decrease,pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" '
                      f'-c:v libx264 -preset ultrafast -crf 18 '
                      f'-c:a aac -b:a 192k -ar 44100 "{out_v.replace("\\", "/")}"')
            subprocess.run(ff_cmd, shell=True, check=True)
        processed_v_paths.append(out_v)
    t2 = time.time() - s2

    # --- PHASE 3: RANDOM ASSEMBLY ---
    s3 = time.time()
    print(f"🎬 [PHASE 3] Generating {len(cut_timestamps)} Rhythmic Segments...")
    v_clips = [VideoFileClip(p) for p in processed_v_paths]
    final_segments = []
    
    for i in range(len(cut_timestamps)):
        t_start = cut_timestamps[i]
        t_end = cut_timestamps[i+1] if i+1 < len(cut_timestamps) else final_audio.duration
        dur = t_end - t_start
        if dur < 0.1: continue
        
        source = random.choice(v_clips)
        v_start = random.uniform(0, max(0, source.duration - dur))
        seg = source.subclip(v_start, v_start + dur).set_start(t_start)
        
        if INC_SRC_AUDIO:
            if seg.audio is not None: seg = seg.volumex(SRC_VOL)
        else:
            seg = seg.without_audio()
            
        final_segments.append(seg)

    if INC_SRC_AUDIO:
        print("🔈 Audio Merge Enabled: Using 'compose' for stability")
        base_video = concatenate_videoclips(final_segments, method="compose")
    else:
        print("🚀 Audio Merge Disabled: Using 'chain' for speed")
        base_video = concatenate_videoclips(final_segments, method="chain")
    t3 = time.time() - s3

    # --- PHASE 4: UI & FX PREP ---
    s4 = time.time()
    print(f"📊 [PHASE 4] Building UI Layers & Spectrogram...")
    y_full, sr_full = librosa.load(temp_wav, sr=22050)
    db_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y_full, sr=sr_full, n_mels=64), ref=np.max)
    if os.path.exists(temp_wav): os.remove(temp_wav)

    prev_h = np.zeros(64)
    t_font = ImageFont.truetype("arialbd.ttf", 28)
    a_font = ImageFont.truetype("arial.ttf", 22)

    def draw_ui(get_frame, t):
        nonlocal prev_h
        global prev_frame_pil
        t = min(t, final_audio.duration - 0.01)
        img = Image.fromarray(get_frame(t))
        
        if M_BLUR > 0 and prev_frame_pil is not None:
            img = Image.blend(img, prev_frame_pil, M_BLUR)
        prev_frame_pil = img.copy()

        if SAT != 1.0: img = ImageEnhance.Color(img).enhance(SAT)
        if CA_AMT > 0:
            r, g, b = img.split()
            img = Image.merge("RGB", (ImageChops.offset(r, int(CA_AMT * 10), 0), g, ImageChops.offset(b, int(-CA_AMT * 10), 0)))
        if BLOOM > 0:
            img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(radius=7)), BLOOM * 0.3)

        # Impact Engine
        zoom, off_x, off_y = 1.0, 0, 0
        if ENABLE_IMPACT:
            eps = 1.0 / FPS 
            recent = [c for c in cut_timestamps if (c-eps) <= t <= (c + IMPACT_DUR)]
            if recent:
                str_val = max(0, 1.0 - ((t - recent[-1]) / IMPACT_DUR))
                zoom = 1.0 + (ZOOM_AMT * str_val)
                s_v = SHAKE_AMT * str_val
                off_x, off_y = random.uniform(-s_v, s_v), random.uniform(-s_v, s_v)

        if zoom > 1.0 or off_x != 0 or off_y != 0:
            img = img.resize((int(TARGET_W * zoom), int(TARGET_H * zoom)), Image.LANCZOS)
            l, tc = (img.width - TARGET_W) / 2 + off_x, (img.height - TARGET_H) / 2 + off_y
            img = img.crop((l, tc, l + TARGET_W, tc + TARGET_H))

        if GRAIN > 0:
            noise = Image.fromarray(np.random.randint(0, 255, (TARGET_H, TARGET_W), dtype='uint8'), mode='L').convert("RGB")
            img = Image.blend(img, noise, GRAIN * 0.15)

        draw = ImageDraw.Draw(img, "RGBA")
        ax, ay = UI_X_OFFSET, UI_Y_OFFSET
        draw.rounded_rectangle([ax, ay, ax + 580, ay + 220], radius=15, fill=(0, 0, 0, 170))
        
        c_idx = int((t / final_audio.duration) * db_spec.shape[1])
        if c_idx < db_spec.shape[1]:
            for i, val in enumerate(db_spec[:, c_idx]):
                h = (int(np.interp(val, [-45, 0], [0, 110])) * 0.3) + (prev_h[i] * 0.7)
                prev_h[i] = h
                if h > 2: draw.rectangle([ax+40+(i*8), ay+155-int(h), ax+40+(i*8)+6, ay+155], fill=(0,255,255,190))

        # Metadata Mapping
        cur_n, l_t, s_dur = audio_files[-1].split('.')[0], t, final_audio.duration
        for s, e, f in song_boundaries:
            if s <= t < e: 
                cur_n, l_t, s_dur = f.split('.')[0], t - s, e - s
                break
        draw.text((ax+550, ay+25), f"{TRACK_PREFIX} {cur_n.replace('_',' ').upper()}", fill="white", font=t_font, anchor="ra")
        draw.text((ax+550, ay+60), ARTIST_NAME.upper(), fill=(180,180,180,255), font=a_font, anchor="ra")
        
        p_ratio = min(l_t / s_dur, 1.0)
        draw.rectangle([ax+35, ay+175, ax+545, ay+181], fill=(40,40,40,255))
        draw.rectangle([ax+35, ay+175, ax+35+int(510*p_ratio), ay+181], fill=(255,0,120,255))
        
        return np.array(img)
    t4 = time.time() - s4

    # --- PHASE 5: MASTER RENDER ---
    s5 = time.time()
    out_name = f"VinCreationz_Master_{datetime.now().strftime('%H%M%S')}.mp4"
    out_path = os.path.join(ROOT, out_name)
    visual_clip = VideoClip(lambda t: draw_ui(base_video.get_frame, t), duration=final_audio.duration)

    if INC_SRC_AUDIO and base_video.audio is not None:
        print("🔊 Merging Audio Streams (Stability Patch Applied)...")
        final_audio_mix = CompositeAudioClip([
            final_audio, 
            base_video.audio.set_start(0)
        ]).set_duration(final_audio.duration)
        final_v = visual_clip.set_audio(final_audio_mix)
    else:
        final_v = visual_clip.set_audio(final_audio)

    print(f"\n🔥 [PHASE 5] Initializing NVENC Master Render: {out_name}")
    final_v.write_videofile(
        out_path, 
        fps=FPS, 
        codec='h264_nvenc', 
        ffmpeg_params=['-rc', 'constqp', '-qp', '23', '-pix_fmt', 'yuv420p'], 
        preset='p4', 
        logger='bar', 
        threads=16,
        audio_codec='libmp3lame',
        audio_fps=44100,
        temp_audiofile="temp-audio.mp3"
    )
    
    # --- PHASE 6: DETAILED SUMMARY ---
    t5 = time.time() - s5
    total_m = (time.time() - total_start) / 60
    
    print("\n" + "="*60)
    print(f"🚀 RENDER COMPLETE IN {total_m:.2f} MINUTES")
    print("="*60)
    print(f"  > AUDIO MIX:  {t1:.2f}s")
    print(f"  > CONFORM:    {t2:.2f}s")
    print(f"  > ASSEMBLY:   {t3:.2f}s")
    print(f"  > UI PREP:    {t4:.2f}s")
    print(f"  > EXPORT:     {t5:.2f}s")
    print("="*60 + "\n")

if __name__ == "__main__":
    create_mix()
