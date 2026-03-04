"""
==============================================================================
VINCREATIONZ LYRIC-SYNC (v1.7) - USER GUIDE & CONFIGURATION
==============================================================================

WHAT THIS DOES:
1. Scans the 'generate_srt' folder for music (MP3/WAV).
2. Uses Whisper AI to "listen" and time-stamp the lyrics.
3. Cleans out AI "hallucinations" (like 'érique' or '[Chorus]' tags).
4. Creates a .LRC file (synced lyrics) for Suno or music players.
5. Moves the finished audio + LRC to the 'test_assets' folder.

DIRECTORY SETUP (Create these before running):
Your_Project/
├── lyric_sync_script.py
└── test_assets/              <-- FINAL OUTPUTS GO HERE
    └── generate_srt/         <-- DROP YOUR RAW FILES HERE

HOW TO USE:
1. Drop your audio file (e.g., "my_song.mp3") into 'test_assets/generate_srt/'.
2. (Optional) Drop a text file with the same name ("my_song.txt") into the 
   same folder. The script uses this to "prime" the AI with the right words.
3. Run the script.
4. Check 'test_assets/' for your synced .mp3 and .lrc files.

CONFIGURATIONS:
- INPUT_FOLDER: Where the script looks for new work.
- DEST_FOLDER:  Where the script "ships" the finished product.
- initial_map:  Uses the first 500 chars of your .txt to help AI accuracy.
- TEMPERATURE:  Set to 0 for "maximum accuracy/no creativity."
==============================================================================
"""
import whisper
import os
import shutil
import re
from datetime import timedelta

INPUT_FOLDER = os.path.join("test_assets", "generate_srt")
DEST_FOLDER = "test_assets"

def format_lrc_time(seconds: float):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02}:{secs:05.2f}]"

def clean_hallucinations(text):
    """Specific filter for the 'érique' bug and other junk."""
    # Strip common Whisper hallucinations
    junk_words = ["érique", "transcription", "subtitles", "thank you for watching"]
    for word in junk_words:
        text = re.sub(word, "", text, flags=re.IGNORECASE)
    
    # Strip Suno brackets
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

def process_sync():
    print(f"--- VinCreationz Lyric-Sync v1.7 (Hallucination Killer) ---")
    
    model = whisper.load_model("medium", device="cuda") 
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".wav", ".mp3"))]

    for filename in files:
        full_input_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        
        lyrics_txt_path = os.path.join(INPUT_FOLDER, f"{base_name}.txt")
        initial_map = ""
        if os.path.exists(lyrics_txt_path):
            with open(lyrics_txt_path, 'r', encoding='utf-8') as f:
                # PRO-TIP: We only feed the first 500 characters as the prompt.
                # If the prompt is too long, Whisper gets confused and ignores the intro!
                initial_map = f.read()[:500] 

        print(f"\n[SYNCING] {filename}...")
        
        result = model.transcribe(
            full_input_path, 
            fp16=True, 
            initial_prompt=initial_map, 
            temperature=0,             
            
            # --- THE SURGEON SETTINGS ---
            compression_ratio_threshold=2.4, # If text starts repeating (érique), it cuts it off.
            logprob_threshold=-1.0,          # Rejects low-confidence 'guesses'.
            no_speech_threshold=0.3,         # Balanced: doesn't skip, but doesn't hallucinate as much.
            condition_on_previous_text=False # Essential: stops one error from ruining the whole song.
        )

        lrc_path = os.path.join(INPUT_FOLDER, f"{base_name}.lrc")
        with open(lrc_path, "w", encoding="utf-8-sig") as l:
            for seg in result['segments']:
                display_text = clean_hallucinations(seg['text'].strip())
                if display_text:
                    timestamp = format_lrc_time(seg['start'])
                    l.write(f"{timestamp}{display_text}\n")

        # Move to Production
        try:
            shutil.move(full_input_path, os.path.join(DEST_FOLDER, filename))
            shutil.move(lrc_path, os.path.join(DEST_FOLDER, f"{base_name}.lrc"))
            print(f"[SUCCESS] Clean sync complete for {base_name}.")
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    process_sync()