#!/usr/bin/env python3

"""
Author: Matthew Tavatgis, Updated by SM, LO 
Created: 27 Jan 2024

Creates a train / test / validation split in the format expected by yolov5+
Optionally trim n images with empty labels
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
import yaml
from statistics import mode
from sklearn.model_selection import StratifiedShuffleSplit

random.seed(1)

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Create train test split for yolov5+')

    parser.add_argument("--src", dest = "src_dir",
            help = "Source directory to parse", default = None, type = str)
    parser.add_argument("--out", dest = "out_dir",
            help = "Output directory for splits (default: same as source)", default = None, type = str)
    parser.add_argument("--valid", dest = "valid",
            help = "Fraction to split for validation, 0-1", default = 0.2, type = float)
    parser.add_argument("--test", dest = "test",
            help = "Fraction to split for testing, 0-1", default = None, type = float)
    parser.add_argument("--dump", dest = "n_dump",
            help = "Number of empty images to drop", default = None, type = int)
    parser.add_argument("--rand", dest = "random_state",
            help = "Seed for random generation", default = 1, type = int)
    parser.add_argument("--min_samples", dest = "min_samples",
            help = "Minimum number of samples per class", default = 10, type = int)

    return parser.parse_args()

def print_progress(current, total, prefix=''):
    """Print progress bar"""
    percent = f"{100 * (current / float(total)):.1f}"
    filled = int(50 * current // total)
    bar = '█' * filled + '-' * (50 - filled)
    print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r')
    if current == total:
        print()

def copy_files_with_progress(src_files, src_dir, dst_dir, total_files, files_copied, desc):
    """Copy files with a single progress bar"""
    for file_name in src_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        copy_file(src_path, dst_path)
        files_copied += 1
        print_progress(files_copied, total_files, f'Copying {desc}')
    return files_copied

def get_class_names(label_dir):
    """Extract unique class names from label files"""
    class_ids = set()
    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
    return sorted(list(class_ids))

def load_original_names(src_dir):
    """Load original class names from source data.yaml"""
    yaml_path = os.path.join(src_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        except Exception as e:
            print(f"Warning: Could not load class names from {yaml_path}: {e}")
    return None

def create_yaml_files(out_dir, class_names, original_names=None):
    """Create data.yaml and test.yaml files, preserving original class names if available"""
    # Use original names if available, otherwise create generic names
    if original_names:
        names = original_names
    else:
        print("\nWarning: No original class names found, using generic class names")
        names = {i: f'class_{i}' for i in class_names}
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(out_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',  # Will be removed if test split is not used
        'names': names
    }
    
    # Write data.yaml
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    # Create test.yaml (copy of data.yaml with val pointing to test)
    test_yaml = data_yaml.copy()
    test_yaml['val'] = 'test/images'
    
    # Write test.yaml
    with open(os.path.join(out_dir, 'test.yaml'), 'w') as f:
        yaml.dump(test_yaml, f, sort_keys=False)
    
    return data_yaml, test_yaml

def main():
    """
    Creates new directory under source and randomly copies specified percentage of images and labels to 'train' or 'valid'
    """
    print("\n📁 Starting dataset split process...")
    
    # Get arguments
    args = arg_parse()
    
    # Set output directory
    out_dir = args.out_dir if args.out_dir is not None else args.src_dir
    
    print(f"\n🔍 Checking source directory: {args.src_dir}")
    print(f"📂 Output directory: {out_dir}")

    # Check source images exist
    image_source_dir = os.path.join(args.src_dir, "all_images")
    if os.path.exists(image_source_dir) == False:
        raise FileNotFoundError(f"❌ Source directory {image_source_dir} does not exist")
    print("✅ Found images directory")

    # Check source labels exist
    label_source_dir = os.path.join(args.src_dir, "all_labels")
    if os.path.exists(image_source_dir) == False:
        raise FileNotFoundError(f"❌ Source directory {label_source_dir} does not exist")
    print("✅ Found labels directory")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("✅ Created output directory")

    # Create output directory paths
    train_out_dir = os.path.join(out_dir, "train")
    valid_out_dir = os.path.join(out_dir, "valid")
    test_out_dir = os.path.join(out_dir, "test")
    train_image_dir = os.path.join(train_out_dir, "images")
    train_label_dir = os.path.join(train_out_dir, "labels")
    valid_image_dir = os.path.join(valid_out_dir, "images")
    valid_label_dir = os.path.join(valid_out_dir, "labels")
    test_image_dir = os.path.join(test_out_dir, "images")
    test_label_dir = os.path.join(test_out_dir, "labels")

    # Prompt to overwrite if output dir exists
    if os.path.exists(train_out_dir) or os.path.exists(valid_out_dir) or os.path.exists(test_out_dir):
        print("\n⚠️  WARNING: Output Directory Exists, data will be overwritten ", end="")

        if input("Y/N?:").lower() != "y":
            print("❌ EXITING...\n")
            exit()
        else:
            print("✅ CONTINUING...\n")
            # Delete existing output dir
            try:
                if os.path.exists(train_out_dir): 
                    shutil.rmtree(train_out_dir)
                    print(f"🗑️  Removed existing {train_out_dir}")
                if os.path.exists(valid_out_dir): 
                    shutil.rmtree(valid_out_dir)
                    print(f"🗑️  Removed existing {valid_out_dir}")
                if os.path.exists(test_out_dir): 
                    shutil.rmtree(test_out_dir)
                    print(f"🗑️  Removed existing {test_out_dir}")
            except OSError as error:
                print(f"❌ Error: {error}")
                sys.exit()

    print("\n📂 Creating output directories...")
    print(f"🚂 TRAIN: {train_out_dir}")
    print(f"✅ VALID: {valid_out_dir}")

    try:
        os.makedirs(train_out_dir)
        os.makedirs(valid_out_dir)
        os.makedirs(train_image_dir)
        os.makedirs(train_label_dir)
        os.makedirs(valid_image_dir)
        os.makedirs(valid_label_dir)
    except OSError as error:
        print(f"❌ Error: {error}")
        sys.exit()

    if args.test is not None:
        print(f"🧪 TEST: {test_out_dir}\n")
        try:
            os.makedirs(test_out_dir)
            os.makedirs(test_image_dir)
            os.makedirs(test_label_dir)
        except OSError as error:
            print(f"❌ Error: {error}")
            sys.exit()
    else:
        print("")

    # Get image paths
    print("📸 Loading images...")
    images = os.listdir(image_source_dir)
    images.sort()
    print(f"✅ Found {len(images)} images")

    # Get label paths
    print("🏷️  Loading labels...")
    labels = os.listdir(label_source_dir)
    labels.sort()
    print(f"✅ Found {len(labels)} labels")
    
    # If dumping empty images
    if args.n_dump is not None:
        print("\n🔍 Checking for empty labels...")
        # Parse labels looking for empty sets to dump
        empty = []
        for i, label in enumerate(labels):
            if os.stat(os.path.join(label_source_dir, label)).st_size == 0:
                empty.append(i)
        
        print(f"Found {len(empty)} empty labels")
        
        # Check numbers
        if args.n_dump > len(empty):
            print("❌ ERROR: Number of empty dumps requested is greater than number of empty labels! EXITING...\n")
            sys.exit()
        
        # Get a random selection of targets to remove
        targets = random.sample(empty, args.n_dump)
        print(f"🗑️  Removing {len(targets)} empty labels...")

        # Purge targets from inputs
        for i in sorted(targets, reverse=True):
            del images[i]
            del labels[i]

    print("\n📊 Analyzing label distribution...")
    # Get label mode for each image
    label_modes = []
    label_counts = {}
    for i, label_name in enumerate(labels):
        print_progress(i + 1, len(labels), 'Processing labels')

        # Get path
        label_path = os.path.join(label_source_dir, label_name)
        
        # Check if empty
        if os.stat(label_path).st_size != 0:
            # If not empty get label mode
            with open(label_path, 'r') as stream:
                lines = stream.readlines()
                
                # Iterate label lines
                cls_list = []
                for label in lines:
                    label = str(label.rstrip()[0])  # Convert to string
                    cls_list.append(label)
                
                # Get mode
                mode_label = mode(cls_list)
                label_modes.append(mode_label)
                # Count occurrences
                if mode_label not in label_counts:
                    label_counts[mode_label] = 0
                label_counts[mode_label] += 1
        else:
            # If empty use '-1' placeholder (as string)
            label_modes.append('-1')
            if '-1' not in label_counts:
                label_counts['-1'] = 0
            label_counts['-1'] += 1
    
    print("\n📊 Label Distribution:")
    # Sort by numeric value after converting to int, but keep as string for display
    for label, count in sorted(label_counts.items(), key=lambda x: int(x[0]) if x[0] != '-1' else -1):
        print(f"- Class {label}: {count} images")
    
    # Remove classes with too few examples
    min_examples = args.min_samples
    problem_classes = {label: count for label, count in label_counts.items() if count < min_examples}
    if problem_classes:
        print(f"\n⚠️  Removing classes with fewer than {min_examples} examples:")
        for label, count in problem_classes.items():
            print(f"- Class {label}: {count} images")
        
        # Create lists of indices to keep and remove
        indices_to_remove = []
        for i, mode_label in enumerate(label_modes):
            if mode_label in problem_classes:
                indices_to_remove.append(i)
        
        # Remove the problematic samples
        for i in sorted(indices_to_remove, reverse=True):
            del images[i]
            del labels[i]
            del label_modes[i]
        
        print(f"\n✅ Removed {len(indices_to_remove)} images with insufficient class samples")
        
        # Update label counts
        label_counts = {}
        for label in label_modes:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        print("\n📊 Updated Label Distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"- Class {label}: {count} images")

    print("\n🔄 Performing stratified split...")
    # Get train test splits
    if args.test is None:
        # Split train and validation sets using stratified (balance preserving split)
        train_images, valid_images, train_labels, valid_labels = set_split(images, labels, label_modes, args.valid)[:4]
        print(f"📈 Split completed: {len(train_images)} train, {len(valid_images)} validation")

    else:
        # Calculate split percentages
        total_per = args.valid + args.test
        test_per = args.test / total_per
        
        # Split train, validation and test sets using stratified (balance preserving split)
        train_images, temp_images, train_labels, temp_labels, temp_modes = set_split(images, labels, label_modes, total_per)
        valid_images, test_images, valid_labels, test_labels = set_split(temp_images, temp_labels, temp_modes, test_per)[:4]
        print(f"📈 Split completed: {len(train_images)} train, {len(valid_images)} validation, {len(test_images)} test")
    
    print("\n📋 Copying files to splits...")
    total_files = len(train_images) * 2 + len(valid_images) * 2 + (len(test_images) * 2 if args.test is not None else 0)
    files_copied = 0

    # Copy train files
    files_copied = copy_files_with_progress(train_images, image_source_dir, train_image_dir, total_files, files_copied, "files")
    files_copied = copy_files_with_progress(train_labels, label_source_dir, train_label_dir, total_files, files_copied, "files")

    # Copy validation files
    files_copied = copy_files_with_progress(valid_images, image_source_dir, valid_image_dir, total_files, files_copied, "files")
    files_copied = copy_files_with_progress(valid_labels, label_source_dir, valid_label_dir, total_files, files_copied, "files")

    # Copy test files if needed
    if args.test is not None:
        files_copied = copy_files_with_progress(test_images, image_source_dir, test_image_dir, total_files, files_copied, "files")
        files_copied = copy_files_with_progress(test_labels, label_source_dir, test_label_dir, total_files, files_copied, "files")

    # Load original class names and create YAML files
    print("\n📝 Creating YAML configuration files...")
    class_names = get_class_names(label_source_dir)
    original_names = load_original_names(args.src_dir)
    data_yaml, test_yaml = create_yaml_files(out_dir, class_names, original_names)
    
    # If no test split, remove test from data.yaml
    if args.test is None:
        data_yaml.pop('test', None)
        with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)

    print("\n✨ Dataset split completed successfully!")
    print(f"""
📊 Final Statistics:
- Training set: {len(train_images)} images/labels
- Validation set: {len(valid_images)} images/labels
{f'- Test set: {len(test_images)} images/labels' if args.test is not None else ''}

📄 YAML Configuration:
- data.yaml: Created for training configuration
- test.yaml: Created for testing configuration
- Number of classes: {len(class_names)}
- Class IDs: {', '.join(map(str, class_names))}
- Class names preserved: {'Yes' if original_names else 'No (using generic names)'}

💡 Note: The YAML files have been configured with absolute paths.
    You can now proceed with training using these YAML files.
    """)

def set_split(src_images, src_labels, src_label_idents, split):
    try:
        # Try stratified split first
        train_split = StratifiedShuffleSplit(n_splits=1, test_size = split, random_state = 1)
        train_gen = train_split.split(np.zeros(len(src_label_idents)), src_label_idents)
        train_index, valid_index = next(train_gen)
    except ValueError:
        # Fall back to random split if stratified fails
        n_samples = len(src_images)
        n_valid = int(n_samples * split)
        indices = np.random.permutation(n_samples)
        valid_index = indices[:n_valid]
        train_index = indices[n_valid:]
    
    # Output arrays
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    valid_idents = []
    
    # Fill 
    for i in train_index:
        train_images.append(src_images[i])
        train_labels.append(src_labels[i])

    for i in valid_index:
        valid_images.append(src_images[i])
        valid_labels.append(src_labels[i])
        valid_idents.append(src_label_idents[i])

    return train_images, valid_images, train_labels, valid_labels, valid_idents

def copy_file(src_dir, dst_dir):
    try:
        shutil.copy(src_dir, dst_dir)
    except OSError as error:
        print(f"\n❌ Error: {error}")
        print(f"\nFailed to copy {src_dir} to {dst_dir}")
        sys.exit()

if __name__=="__main__":
    main()
