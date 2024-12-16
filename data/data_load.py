import json
import pandas as pd
import os
from collections import defaultdict


class DataLoader:
    def __init__(self, folder, file_name, max_users, min_rows_per_user, batch_size=1000):
        self.folder = folder
        self.file_name = file_name
        self.max_users = max_users
        self.min_rows_per_user = min_rows_per_user
        self.meta_filename = 'meta_' + self.file_name
        self.batch_size = batch_size

    def read_jsonl(self, file_path):
        """Reads a JSONL file and returns a list of dictionaries."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def load_meta_data(self):
        """Loads metadata from the meta file and returns a mapping of parent_asin to title."""
        meta_path = os.path.join(self.folder, self.meta_filename)
        meta_data = self.read_jsonl(meta_path)
        meta_mapping = {entry["parent_asin"]: entry.get("title", "Unknown") for entry in meta_data}
        return meta_mapping

    def process_data(self):
        """Processes the main JSONL file and filters user data based on criteria."""
        file_path = os.path.join(self.folder, self.file_name)
        user_data = defaultdict(list)

        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                batch_data = []
                for _ in range(self.batch_size):
                    line = f.readline()
                    if not line:
                        break
                    entry = json.loads(line.strip())
                    if entry.get("verified_purchase") is True:
                        user_id = entry.get("user_id")
                        if user_id:
                            batch_data.append({
                                "parent_asin": entry.get("parent_asin"),
                                "user_id": user_id,
                                "timestamp": entry.get("timestamp"),
                            })
                if batch_data:
                    for entry in batch_data:
                        user_data[entry["user_id"]].append(entry)
                if not line:
                    break

        selected_data = []
        for user_id, entries in user_data.items():
            if len(entries) >= self.min_rows_per_user:
                selected_data.extend(entries[:self.min_rows_per_user])
                if len(selected_data) >= self.max_users * self.min_rows_per_user:
                    break

        return selected_data

    def save_loaded_data(self):
        """Saves the processed data with titles to a CSV file."""
        data = self.process_data()
        meta_mapping = self.load_meta_data()

        # Add title information to each entry
        for entry in data:
            entry["title"] = meta_mapping.get(entry["parent_asin"], "Unknown")

        df = pd.DataFrame(data)

        output_name = os.path.basename(self.file_name).replace(".jsonl", ".csv")
        output_file = os.path.join(self.folder, output_name)

        df.to_csv(output_file, index=False)
        print(f"Data saved to: {output_file}")

        return output_file
