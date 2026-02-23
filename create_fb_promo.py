"""
SCRIPT: VinCreationz FB Promo Generator (Standalone)
PURPOSE: Creates 15s viral loops with text overlays for FB/Reels.
USAGE: Configure 'FB_promo_config.txt' and run.
"""

import os
import configparser
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, vfx
from moviepy.config import change_settings

# Point to ImageMagick (Required for TextClip)
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})

CONFIG_FILE = "FB_promo_config.txt"

def parse_time(time_str):
    """Converts '00:21' or '1:05' to seconds."""
    try:
        if ":" in str(time_str):
            m, s = time_str.split(":")
            return int(m) * 60 + float(s)
        return float(time_str)
    except:
        return 0.0

def create_promo():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Error: {CONFIG_FILE} not found.")
        return

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE, encoding='utf-8')

    # Global Settings
    output_folder = config.get("Global", "output_folder", fallback="FB_Promos")
    font = config.get("Global", "font_path", fallback="arial.ttf")
    burn_text = config.getboolean("Global", "burn_text", fallback=True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each song section
    for section in config.sections():
        if section == "Global": continue
        
        print(f"\n🎬 Processing: {section}...")
        
        try:
            # 1. Load Paths
            audio_path = config.get(section, "audio_file")
            video_path = config.get(section, "video_loop")
            
            if not os.path.exists(audio_path) or not os.path.exists(video_path):
                print(f"   ⚠️ Skipping: File not found ({audio_path} or {video_path})")
                continue

            # 2. Settings
            start_time = parse_time(config.get(section, "start_time"))
            duration = config.getfloat(section, "duration", fallback=15.0)
            top_txt_str = config.get(section, "top_text", fallback="")
            bot_txt_str = config.get(section, "bottom_text", fallback="")

            # 3. Audio Processing
            audio = AudioFileClip(audio_path)
            # Safety check for duration
            if start_time + duration > audio.duration:
                print("   ⚠️ Warning: Start time + Duration exceeds song length. Trimming to end.")
                duration = audio.duration - start_time
            
            audio_cut = audio.subclip(start_time, start_time + duration)

            # 4. Video Loop Processing
            video = VideoFileClip(video_path)
            # Resize to Vertical (9:16) if not already
            # Logic: Zoom to fill 1080x1920
            target_ratio = 1080 / 1920
            current_ratio = video.w / video.h
            
            if current_ratio > target_ratio:
                # Video is too wide -> Crop width
                new_w = int(video.h * target_ratio)
                crop_x = (video.w - new_w) // 2
                video = video.crop(x1=crop_x, width=new_w)
            else:
                # Video is too tall -> Crop height
                new_h = int(video.w / target_ratio)
                crop_y = (video.h - new_h) // 2
                video = video.crop(y1=crop_y, height=new_h)
                
            video = video.resize((1080, 1920))
            
            # Loop video to match audio duration
            final_clip = video.loop(duration=duration)
            final_clip = final_clip.set_audio(audio_cut)

            # 5. Text Overlays (Optional)
            layers = [final_clip]
            
            if burn_text:
                if top_txt_str:
                    # Top Text (White with Black Stroke)
                    txt_top = TextClip(top_txt_str, font=font, fontsize=90, color='white', 
                                     stroke_color='black', stroke_width=4, size=(1000, None), method='caption')
                    txt_top = txt_top.set_position(('center', 200)).set_duration(duration)
                    layers.append(txt_top)

                if bot_txt_str:
                    # Bottom Text (Yellow call to action)
                    txt_bot = TextClip(bot_txt_str, font=font, fontsize=70, color='yellow', 
                                     stroke_color='black', stroke_width=3, size=(1000, None), method='caption')
                    txt_bot = txt_bot.set_position(('center', 1500)).set_duration(duration)
                    layers.append(txt_bot)

            # 6. Render
            final = CompositeVideoClip(layers, size=(1080, 1920))
            output_filename = f"{section.replace(' ', '_')}_FB_Promo.mp4"
            final.write_videofile(os.path.join(output_folder, output_filename), 
                                fps=30, codec='libx264', audio_codec='aac', 
                                threads=4, preset='fast')
            
            print(f"✅ Created: {output_filename}")

        except Exception as e:
            print(f"❌ Error processing {section}: {e}")

if __name__ == "__main__":
    create_promo()