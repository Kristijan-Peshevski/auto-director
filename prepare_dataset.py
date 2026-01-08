import os
import time
import json
import google.generativeai as genai
import ffmpeg

# ================= CONFIGURATION =================
# 1. PASTE YOUR API KEY HERE
API_KEY = ""

# 2. FOLDER SETUP
# Get the directory where this script is currently running
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# INPUT: Look for videos in the SAME folder as this script
INPUT_FOLDER = CURRENT_DIR

# OUTPUT: Create a subfolder inside the current directory
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "mr_beast_dataset")
# =================================================

# Configure Google AI
genai.configure(api_key=API_KEY)


def analyze_and_cut(video_path, filename):
    print(f"Processing: {filename}...")

    try:
        # --- STEP 1: UPLOAD TO GOOGLE ---
        print("  [1/3] Uploading video to Gemini...")
        video_file = genai.upload_file(path=video_path)

        # Wait for processing to complete
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            print(f"  ERROR: Google failed to process video. State: {video_file.state.name}")
            return

        # --- STEP 2: ANALYZE VIDEO ---
        print("  [2/3] Analyzing with Gemini 3 Flash...")

        prompt = """
        Analyze this video for an AI training dataset.
        Find 5 to 8 distinct, high-energy, visually interesting moments (explosions, stunts, fast motion, money).
        Ignore static talking heads.

        Return a RAW JSON list.
        Format: [{"start": "MM:SS", "end": "MM:SS", "caption": "visual description"}]

        Constraints:
        1. Clips must be between 2 and 4 seconds long.
        2. Do NOT use the name "MrBeast". Use "a man" or "a person".
        3. Describe the visual style (e.g., "cinematic lighting", "wide shot", "color graded").
        4. Output ONLY valid JSON. No markdown formatting.
        """

        # *** UPDATED TO YOUR AVAILABLE MODEL ***
        model = genai.GenerativeModel(model_name="gemini-3-flash-preview")

        response = model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 600}
        )

        # Clean up the response
        text_resp = response.text
        if "```json" in text_resp:
            text_resp = text_resp.replace("```json", "").replace("```", "")
        text_resp = text_resp.strip()

        # Parse JSON
        clips = json.loads(text_resp)

        # --- STEP 3: CUT VIDEO ---
        print(f"  [3/3] Cutting {len(clips)} clips...")
        base_name = os.path.splitext(filename)[0]

        for i, clip in enumerate(clips):
            # Helper to convert MM:SS to seconds
            def to_sec(ts):
                parts = list(map(int, ts.split(':')))
                return parts[0] * 60 + parts[1]

            start = to_sec(clip['start'])
            end = to_sec(clip['end'])
            duration = end - start

            # Enforce max duration of 4 seconds
            if duration > 4: duration = 4

            # Define output filenames
            out_name = f"{base_name}_{i}"
            vid_out = os.path.join(OUTPUT_FOLDER, f"{out_name}.mp4")
            txt_out = os.path.join(OUTPUT_FOLDER, f"{out_name}.txt")

            # Run FFMPEG
            try:
                (
                    ffmpeg
                    .input(video_path, ss=start, t=duration)
                    .output(vid_out, an=None, vcodec='libx264', crf=18)
                    .run(quiet=True, overwrite_output=True)
                )

                # Save the text caption
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(f"mrbeast style, {clip['caption']}")

                print(f"    -> Saved: {out_name}.mp4")

            except ffmpeg.Error as e:
                print(f"    FFmpeg Error: {e}")

    except Exception as e:
        print(f"  FAILED to process {filename}: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    print(f"Looking for videos in: {INPUT_FOLDER}")
    print("Starting Auto-Director...")

    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
            # Skip the output folder itself
            if file == "mr_beast_dataset": continue

            full_path = os.path.join(INPUT_FOLDER, file)
            analyze_and_cut(full_path, file)

            # Increased sleep time to 5 seconds to be safe with the Preview model
            time.sleep(5)

    print("\nDone! Check the 'mr_beast_dataset' folder.")