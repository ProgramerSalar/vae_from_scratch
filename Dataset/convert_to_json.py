import csv
import json 
import os 


INPUT_CSV_FILE = "../../vae_from_scratch/annotation/OpenVid-1M.csv"
OUTPUT_JSON_FILE = '../../vae_from_scratch/annotation/openVid-1M.json'

def convert_csv_to_json(csv_path, json_path):

    # check if the input csv file exists 
    if not os.path.exists(csv_path):
        # os.makedirs(csv_path, exist_ok=True)
        print("Please Insert the csv file")

    json_data_list = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:

            csv_reader = csv.DictReader(csv_file)
            # check if neccearry column are exist.
            if 'video' not in csv_reader.fieldnames or 'caption' not in csv_reader.fieldnames:
                raise FileExistsError("so your csv file doesn't found the fieldName `video` or 'caption` ")
            
            # Loop through each row in the CSV file 
            for row in csv_reader:
                # create a new dict in the target format.
                data_entry = {
                    "video": row.get("video", ""),
                    "text": row.get("caption", ""),
                    "latent": "",
                    "text_fea": ""
                }
                # Add our new dictionary to the lit 
                json_data_list.append(data_entry)
                
        # After processing all rows, write the complete list to the JSON file.
        with open(json_path, mode="w", encoding="utf-8") as json_file:
            json.dump(json_data_list, json_file, indent=4)

        print(f"Successfully dumps the `json` into file: {json_path}")


    except FileNotFoundError("File not found."):
        print("Please Insert csv file path")











if __name__ == "__main__":
    out = convert_csv_to_json(csv_path=INPUT_CSV_FILE,
                              json_path=OUTPUT_JSON_FILE)