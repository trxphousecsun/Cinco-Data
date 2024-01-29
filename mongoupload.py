import os
import json
import concurrent.futures
from pymongo import MongoClient
from bson import json_util, ObjectId

# MongoDB connection specifics
mongo_uri = "mongodb://localhost:27017/"
db_name = "history"  # Replace with your database name
collection_name = "history"  # Replace with your collection name

# Directory containing the JSON files
directory_path = "/Users/jazzhashzzz/Documents/submissions"

def split_large_document(doc, common_id):
    max_size = 16 * 1024 * 1024  # 16 MB
    parts = []
    part_number = 1

    current_part = {"common_id": common_id, "part_number": part_number}
    current_size = len(json_util.dumps(current_part))

    for key, value in doc.items():
        item_size = len(json_util.dumps({key: value}))
        if current_size + item_size > max_size:
            parts.append(current_part)
            part_number += 1
            current_part = {"common_id": common_id, "part_number": part_number}
            current_size = len(json_util.dumps(current_part))

        current_part[key] = value
        current_size += item_size

    if current_part:
        parts.append(current_part)

    return parts

def upload_file(file_path, collection):
    with open(file_path, 'r') as file:
        file_data = json.load(file)

        common_id = ObjectId()
        if len(json_util.dumps(file_data)) > 16 * 1024 * 1024:
            parts = split_large_document(file_data, common_id)
            collection.insert_many(parts)
            print(f"Uploaded {len(parts)} parts of {file_path} to MongoDB.")
        else:
            file_data["common_id"] = common_id
            file_data["part_number"] = 1
            collection.insert_one(file_data)
            print(f"Uploaded {file_path} to MongoDB.")

def upload_files_to_mongodb():
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    files_to_upload = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            files_to_upload.append(file_path)

    batch_size = 1000  # Adjust the batch size as needed
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(files_to_upload), batch_size):
            batch = files_to_upload[i:i+batch_size]
            executor.map(upload_file, batch, [collection] * len(batch))

    print("All files uploaded successfully.")

if __name__ == "__main__":
    upload_files_to_mongodb()