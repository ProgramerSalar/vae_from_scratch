import json 
import os 

INPUT_JSON_FILE = '/home/manish/Desktop/projects/video_vae/vae_from_scratch/annotation/openVid-1M.json'
VIDEO_DIR = "/home/manish/Dataset/video_dataset/OpenVid_part0/OpenVid_part1"
OUTPUT_JSON_FILE = "../../vae_from_scratch/annotation/OpenVid-1M_verify_video_data.json"
prefix_remove = "/home/manish/"
repace_prefix = "../../"

def verify_video_files():

    if not os.path.exists(INPUT_JSON_FILE):
        print("Please provide the `INPUT_JSON_FILE` path")

    if not os.path.exists(VIDEO_DIR):
        print("Please provide the `VIDEO_DIR` path")

    if not os.path.exists(OUTPUT_JSON_FILE):
        print("Please provide the `OUTPUT_JSON_FILE` path.")

    try:

        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("json data does not found list format.")

    except FileNotFoundError:
        print("Please provide the json file")

    # verify files 
    
    found_entries = []
    missing_files = []
    for entry in data:
        # Get the video path form the JSON 
        video_path_from_json = entry.get('video')
        if not video_path_from_json:
            print(f"There is no any path are found video")
            continue

        video_filename = os.path.basename(video_path_from_json)
        full_video_path_to_check = os.path.join(VIDEO_DIR, video_filename)
        update_video_path = full_video_path_to_check.replace(prefix_remove, repace_prefix)
        # print(update_video_path)

        data_entry = {
            "video": update_video_path,
            "text": entry.get("text", ""),
            "latent": "",
            "text_fea": ""
        }
        
        if os.path.exists(full_video_path_to_check):
            found_entries.append(data_entry)
        else:
            missing_files.append(video_filename)

        

    print(f"Found {len(found_entries)} matching video files")
    print(f"Missing {len(missing_files)} video files.")

    



    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(found_entries, f, indent=4)

    except FileNotFoundError("Please provide `OUTPUT_JSON_FILE`"):
        print("file not foud.")



        
    
    

if __name__ == "__main__":
    out = verify_video_files()
    print(out)