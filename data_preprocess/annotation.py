import json
from typing import List
import os

def fileter_annotation(desired_labels: List[str], annotation_path: str,output_file_path: str):



    # Find the original indices of the desired labels
    

    # Create a mapping from original COCO indices to new indices (keeping their order relative to their original index)
    

    # Load COCO annotation file
    with open(annotation_path, "r") as f:
        data = json.load(f)

    label_to_original_index  = {}

    for c in data["categories"]:
        label_to_original_index[c["name"]] = c["id"]  # Map the original COCO index to its corresponding label name
    desired_indices = set([label_to_original_index[label.replace("_", " ")] for label in desired_labels])

    # Filter only annotations belonging to desired categories and reassign indices
    filtered_annotations = []
 
    for ann in data['annotations']:
        if ann['category_id'] in desired_indices:
            filtered_annotations.append(ann)

    # Filter the categories list to only include the desired categories with their new indices
    filtered_categories = []
    for c in data["categories"]:
        if c['id'] in desired_indices:
            filtered_categories.append(c)

    # Create new JSON structure with filtered annotations and categories
    filtered_data = {
        "images": data["images"],  # Keep original image information
        "annotations": filtered_annotations,
        "categories": filtered_categories
    }

    if not os.path.exists(output_file_path):
        if os.path.dirname(output_file_path) != "" and os.path.dirname(output_file_path) != ".":
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Create directory if it doesn't exist
    # Write to a new JSON file
    with open(output_file_path, "w") as f:
        json.dump(filtered_data, f)

    print("Filtered annotations saved to 'filtered_annotations.json'")
