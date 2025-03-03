#!/usr/bin/env python3

"""
Author: Lauren Olinger
Created: March 2025

===
Generates a clean class listing format from COCO instances.json for class merging
Can optionally pre-fill mappings from an existing Families_class_changes file
Classes can be sorted by ID (default) or alphabetically by name
===
"""

import json
import argparse
import os
import sys
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

def find_instances_json(directory: str) -> str:
    """Find instances_default.json in the annotations subfolder"""
    expected_path = os.path.join(directory, "annotations", "instances_default.json")
    if not os.path.exists(expected_path):
        sys.exit(f"Error: Could not find instances_default.json at {expected_path}")
    return expected_path

def parse_class_changes(file_path: str) -> Dict[str, str]:
    """Parse class mappings from a Families_class_changes file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find the class_change dictionary using regex
        dict_match = re.search(r'class_change\s*=\s*{([^}]+)}', content, re.DOTALL)
        if not dict_match:
            sys.exit("Could not find class_change dictionary in the file")
            
        dict_str = dict_match.group(1)
        
        # Parse the mappings
        mappings = {}
        for line in dict_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Extract the key-value pair
            match = re.match(r"'(\d+)'\s*:\s*'(\d+|9)'", line)
            if match:
                old_class, new_class = match.groups()
                mappings[old_class] = new_class
                
        return mappings
        
    except Exception as e:
        sys.exit(f"Error parsing Families_class_changes file: {e}")

def load_instances_json(json_path: str) -> Dict:
    """Load and parse the COCO instances.json file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        sys.exit(f"Error loading instances.json: {e}")

def extract_categories(data: Dict, sort_by_name: bool = False) -> List[Dict]:
    """
    Extract category information from instances.json
    
    Args:
        data: The loaded JSON data
        sort_by_name: If True, sort categories alphabetically by name instead of by ID
    """
    try:
        categories = data['categories']
        # Sort either by ID (default) or by name
        if sort_by_name:
            return sorted(categories, key=lambda x: x['name'].lower())
        else:
            return sorted(categories, key=lambda x: x['id'])
    except KeyError:
        sys.exit("Error: instances.json does not contain 'categories' section")

def generate_class_table(categories: List[Dict], output_path: str, existing_mappings: Optional[Dict[str, str]] = None):
    """Generate a clean text file with class information and mapping template"""
    
    # Calculate maximum lengths for formatting
    max_id_len = max(len(str(cat['id'])) for cat in categories)
    max_name_len = max(len(cat['name']) for cat in categories)
    max_id_len = max(max_id_len, len("Class ID"))
    max_name_len = max(max_name_len, len("Current Class Name"))
    
    # Create header
    header = f"""# Class Mapping Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# Instructions:
# 1. Review the current classes below
# 2. In the "Map To Class" column:
#    - Enter the new class ID to merge this class into
#    - Leave blank to keep the class as is
#    - Use 'remove' to exclude this class
# 3. In the "New Class Label" column:
#    - Enter a descriptive label for any new class IDs used
#    - This helps document what each new class represents
#
# Format:
# {'-' * (max_id_len + max_name_len + 30)}
# {'Class ID'.ljust(max_id_len)} | {'Current Class Name'.ljust(max_name_len)} | Map To Class | New Class Label
# {'-' * (max_id_len + max_name_len + 30)}"""

    # Create class listing
    rows = []
    for cat in categories:
        class_id = str(cat['id'])
        # Get existing mapping if available
        mapping = existing_mappings.get(class_id, '__________') if existing_mappings else '__________'
        # Convert '9' to 'remove' for better clarity
        mapping = 'remove' if mapping == '9' else mapping
        row = f"{class_id.ljust(max_id_len)} | {cat['name'].ljust(max_name_len)} | {mapping.ljust(10)} | __________"
        rows.append(row)

    # Add footer with examples and note about existing mappings
    footer = f"""
# {'-' * (max_id_len + max_name_len + 30)}
#
# Example mappings:
# Class ID | Current Class Name     | Map To Class | New Class Label
# 0        | dog                    | 1            | animals
# 1        | cat                    | 1            | animals
# 2        | person                 | 2            | people
# 3        | bicycle               | remove        | 
#
# The above example:
# - Merges 'dog' and 'cat' into new class 1 labeled 'animals'
# - Maps 'person' to new class 2 labeled 'people'
# - Removes the 'bicycle' class
"""

    if existing_mappings:
        footer += """
# Note: Existing mappings from Families_class_changes have been pre-filled.
# Please review these mappings and add appropriate class labels.
"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))
        f.write(footer)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a clean class listing format from COCO instances_default.json')
    
    parser.add_argument('--dir', 
                       help='Directory containing annotations/instances_default.json',
                       required=True)
    
    parser.add_argument('--output',
                       help='Path to output text file (default: class_mapping.txt)',
                       default='class_mapping.txt')
    
    parser.add_argument('--existing-mappings',
                       help='Path to existing Families_class_changes file to pre-fill mappings',
                       default=None)
    
    parser.add_argument('--sort-by-name',
                       help='Sort classes alphabetically by name instead of by ID',
                       action='store_true')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find and load instances_default.json
    json_path = find_instances_json(args.dir)
    print(f"Found instances_default.json at: {json_path}")
    
    # Load existing mappings if provided
    existing_mappings = None
    if args.existing_mappings:
        print(f"Loading existing mappings from: {args.existing_mappings}")
        existing_mappings = parse_class_changes(args.existing_mappings)
        print(f"Found {len(existing_mappings)} existing mappings")
    
    # Load and process data
    data = load_instances_json(json_path)
    categories = extract_categories(data, args.sort_by_name)
    
    # Generate output
    generate_class_table(categories, args.output, existing_mappings)
    print(f"Generated class mapping template at: {args.output}")
    if existing_mappings:
        print("Existing mappings have been pre-filled. Please add class labels to complete the mapping.")
    else:
        print("Edit this file to specify your class mappings")

if __name__ == '__main__':
    main() 