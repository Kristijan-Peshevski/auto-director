import os
import csv

# Point this to your dataset folder
DATASET_DIR = "/workspace/training_dataset"
CSV_PATH = os.path.join(DATASET_DIR, "metadata.csv")

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write the headers that your command expects
    writer.writerow(["video", "prompt"])

    # Loop through files
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".mp4"):
            video_path = filename  # Relative path is fine if csv is in same folder
            txt_name = filename.replace(".mp4", ".txt")

            # Read the caption
            if os.path.exists(os.path.join(DATASET_DIR, txt_name)):
                with open(os.path.join(DATASET_DIR, txt_name), 'r') as txt_file:
                    caption = txt_file.read().strip()

                writer.writerow([video_path, caption])
                print(f"Added: {filename}")

print(f"\nSUCCESS! metadata.csv created at {CSV_PATH}")