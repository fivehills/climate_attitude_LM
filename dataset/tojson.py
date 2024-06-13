import csv
import json

# Define the path to your CSV file
csv_file_path = 'test.csv'

# Define the path where the JSON file will be saved
json_file_path = 'test.json'

# Initialize a list to hold the JSON entries
json_data = []

# Open the CSV file for reading
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    # Create a CSV reader object
    reader = csv.DictReader(csvfile)  # Default delimiter is comma
    for row in reader:
        # Convert label number to string
        label_map = { '0': 'risk', '1': 'neutral', '2': 'opportunity' }
        label_text = label_map.get(row['label'], 'unknown')  # Default to 'unknown' if label is not 0, 1, or 2

        # Create a dictionary for each row
        entry = {
            "instruction": row['text'],
            "input": "",
            "output": label_text
        }
        # Append the dictionary to the list
        json_data.append(entry)

# Convert the list to JSON format
json_output = json.dumps(json_data, indent=4)

# Save the JSON output to a file
with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    jsonfile.write(json_output)

# Print confirmation that the file has been saved
print(f"JSON data has been saved to {json_file_path}")

