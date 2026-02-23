import ee
import geemap
import os
import pandas as pd
import subprocess
from datetime import datetime
from PIL import Image, ImageDraw

# --- START TELEMETRY ---
start_time = datetime.now()
print(f"🚀 Execution Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Initialize
project_id = 'surface-temp-trend' 
try:
    ee.Initialize(project=project_id)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=project_id)

roi = ee.Geometry.Rectangle([-85.35, 35.00, -85.15, 35.18])
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'Chatt_Pro_Pulse')
if not os.path.exists(out_dir): os.makedirs(out_dir)

# --- CONFIG: POINT THIS TO YOUR ACTUAL FFMPEG.EXE PATH ---
# Example: r'E:\Tools\ffmpeg\bin\ffmpeg.exe'
ffmpeg_path = 'ffmpeg' # Change this to the full path if it keeps failing

# 2. Thermal Function with Gap Healing
def get_healed_monthly(year, month):
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(roi).filterDate(start_date, end_date))
    
    if col.size().getInfo() == 0: return None

    def apply_scale(img):
        return img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    primary = ee.Image(col.map(apply_scale).median()).clip(roi)
    fallback = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(roi)
                .filter(ee.Filter.calendarRange(month, month, 'month'))
                .filter(ee.Filter.calendarRange(year-2, year+2, 'year'))
                .map(apply_scale).median().clip(roi))

    return primary.unmask(fallback).set('year', year).set('month', month)

# 3. Processing Loop (Skips what's already done)
years, months = range(2014, 2026), range(1, 13)
# --- REVISED FAST-SKIP LOOP ---
for yr in years:
    for mo in months:
        png_path = os.path.join(out_dir, f"frame_{yr}_{mo:02d}.png")
        
        # 1. IMMEDIATE LOCAL CHECK (Saves several seconds per month)
        if os.path.exists(png_path):
            # We already have the labeled PNG, skip everything else
            continue

        # 2. ONLY CALL GOOGLE IF FILE IS MISSING
        print(f"🛰️ Requesting new data from GEE for {yr}-{mo:02d}...")
        img = get_healed_monthly(yr, mo)
        
        if img is None:
            print(f"⏩ No data for {yr}-{mo:02d}, skipping.")
            continue
            
        # 3. GENERATE AND SAVE
        vis = {'min': 0, 'max': 40, 'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']}
        geemap.get_image_thumbnail(img, png_path, vis, dimensions=768, region=roi)
        
        # 4. LABEL AND PERMANENTLY STORE
        label_img = Image.open(png_path).convert('RGB')
        draw = ImageDraw.Draw(label_img)
        draw.text((30, 30), f"{yr}-{mo:02d}", fill="white")
        label_img.save(png_path)
        print(f"✅ Rendered and Saved: {yr}-{mo:02d}")

# 4. GPU-Accelerated FFmpeg Blend
output_mp4 = os.path.join(out_dir, "hixson_thermal_pulse.mp4")

if os.path.exists(output_mp4):
    print(f"✅ Video already exists at {output_mp4}. Delete it if you want to re-render.")
else:
    print("\n🎬 Blending frames using RTX 4070 (NVENC)...")
    # Using 'minterpolate' to turn 12fps into 60fps liquid motion
    ffmpeg_cmd = [
        ffmpeg_path, '-y',
        '-framerate', '12',
        '-pattern_type', 'glob', '-i', os.path.join(out_dir, "frame_*.png"),
        '-vf', 'minterpolate=fps=60:mi_mode=blend', 
        '-c:v', 'h264_nvenc', 
        '-preset', 'p7', 
        output_mp4
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"🔥 VIDEO COMPLETE: {output_mp4}")
    except FileNotFoundError:
        print(f"❌ ERROR: FFmpeg not found. Please update 'ffmpeg_path' in the script.")
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")

# --- END TELEMETRY ---
end_time = datetime.now()
print(f"\n🏁 Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Total Runtime: {end_time - start_time}")