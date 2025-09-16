import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import sys
import argparse

class TextGridParser:
    """Custom parser for Praat TextGrid files"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.tiers = {}
        self.parse()
    
    def parse(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse basic information
        self.xmin = float(re.search(r'xmin\s*=\s*([\d.]+)', content).group(1))
        self.xmax = float(re.search(r'xmax\s*=\s*([\d.]+)', content).group(1))
        
        # Find all tiers
        tier_pattern = r'item\s*\[(\d+)\]:\s*class\s*=\s*"(\w+)"\s*name\s*=\s*"([^"]+)"(.*?)(?=item\s*\[\d+\]:|$)'
        tier_matches = re.findall(tier_pattern, content, re.DOTALL)
        
        for tier_idx, tier_class, tier_name, tier_content in tier_matches:
            if tier_class == "IntervalTier":
                intervals = self.parse_intervals(tier_content)
                self.tiers[tier_name] = intervals
    
    def parse_intervals(self, content):
        intervals = []
        interval_pattern = r'intervals\s*\[(\d+)\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"'
        matches = re.findall(interval_pattern, content)
        
        for idx, xmin, xmax, text in matches:
            intervals.append({
                'xmin': float(xmin),
                'xmax': float(xmax),
                'text': text.strip()
            })
        
        return intervals
    
    def __getitem__(self, key):
        return self.tiers.get(key, [])

def extract_feature_values(npy_data, start_time, end_time, sampling_rate=100):
    """
    Extract all feature values for a given time range.
    
    Args:
        npy_data: Numpy array with features
        start_time: Start time in seconds
        end_time: End time in seconds
        sampling_rate: Sampling rate in Hz
    
    Returns:
        dict: Dictionary with feature names as keys and lists of values as values
    """
    start_frame = int(start_time * sampling_rate)
    end_frame = int(end_time * sampling_rate)
    
    # Ensure frames are within bounds
    start_frame = max(0, min(start_frame, npy_data.shape[1] - 1))
    end_frame = max(0, min(end_frame, npy_data.shape[1] - 1))
    
    feature_values = {}
    feature_names = ["LA", "LP", "TBCL", "TBCD", "TTCL", "TTCD"]
    
    for feature_idx, feature_name in enumerate(feature_names):
        if start_frame <= end_frame:
            values = npy_data[feature_idx, start_frame:end_frame + 1].tolist()
        else:
            values = []
        feature_values[feature_name] = values
    
    return feature_values

def calculate_feature_averages(feature_values):
    """
    Calculate average values for each feature.
    
    Args:
        feature_values: Dictionary with feature names and lists of values
    
    Returns:
        dict: Dictionary with average values for each feature
    """
    averages = {}
    for feature_name, values in feature_values.items():
        if values:
            averages[feature_name] = np.mean(values)
        else:
            averages[feature_name] = np.nan
    return averages

def process_middle_phonemes(folder_a, folder_b, target_phonemes):
    """
    Process phonemes as middle phonemes from TextGrid files and extract corresponding features from npy files.
    
    Args:
        folder_a (str): Path to folder containing npy files
        folder_b (str): Path to folder containing subfolders with TextGrid and wav files
        target_phonemes (list): List of target phonemes to search for as middle phonemes (e.g., ['ʃ', 's', 't͡ʃ'])
    
    Returns:
        pd.DataFrame: DataFrame containing all extracted features and metadata
    """
    
    # List to store all results
    results = []
    
    # Get all npy files from folder A
    npy_files = list(Path(folder_a).glob("*.npy"))
    print(f"Found {len(npy_files)} npy files in folder A")
    
    for npy_path in npy_files:
        # Extract base filename without extension
        base_name = npy_path.stem
        
        # Remove potential suffixes like '_predict'
        if '_predict' in base_name:
            base_name = base_name.replace('_predict', '')
        if '_xrmb_tv' in base_name:
            base_name = base_name.replace('_xrmb_tv', '')
        
        # Find corresponding TextGrid file in folder B
        textgrid_path = None
        wav_path = None
        
        # Search through all subdirectories in folder B
        for root, dirs, files in os.walk(folder_b):
            for file in files:
                if file.startswith(base_name) and file.endswith('.TextGrid'):
                    textgrid_path = os.path.join(root, file)
                elif file.startswith(base_name) and file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
        
        if not textgrid_path:
            print(f"Warning: Could not find matching TextGrid file for {base_name}")
            continue
        if not wav_path:
            print(f"Warning: Could not find matching wav file for {base_name}")
            continue
        
        print(f"Processing: {base_name}")
        
        try:
            # Load TextGrid file with custom parser
            tg = TextGridParser(textgrid_path)
            
            # Load npy data
            npy_data = np.load(npy_path)
            print(f"  Loaded npy data with shape: {npy_data.shape}")
            
            # Process each target phoneme as middle phoneme
            for phoneme in target_phonemes:
                phoneme_results = find_middle_phoneme(
                    tg, npy_data, base_name, wav_path, phoneme
                )
                if phoneme_results:
                    print(f"  Found {len(phoneme_results)} instances of {phoneme} as middle phoneme")
                    results.extend(phoneme_results)
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def find_middle_phoneme(tg, npy_data, base_name, wav_path, target_phoneme):
    """
    Find a specific phoneme as middle phoneme in TextGrid and extract corresponding features.
    Skip instances where the phoneme is adjacent to silence ('sil') or at utterance boundaries.
    
    Args:
        tg: TextGrid object
        npy_data: Numpy array with features
        base_name: Base filename
        wav_path: Path to wav file
        target_phoneme: Target phoneme to find as middle phoneme
    
    Returns:
        list: List of dictionaries with extracted data
    """
    
    results = []
    
    # Get phones and words tiers
    phones_tier = tg['phones'] if 'phones' in tg.tiers else []
    words_tier = tg['words'] if 'words' in tg.tiers else []
    
    if not phones_tier:
        return results
    
    # Find all occurrences of the target phoneme
    for i in range(len(phones_tier)):
        current_phoneme = phones_tier[i]['text'].strip().lower()
        
        # Check if current phoneme matches target phoneme
        if current_phoneme == target_phoneme.lower():
            # Skip if previous or next phoneme is 'sil' (silence)
            if i > 0 and phones_tier[i-1]['text'].strip().lower() == 'sil':
                continue
            if i < len(phones_tier) - 1 and phones_tier[i+1]['text'].strip().lower() == 'sil':
                continue
            
            # Skip if at utterance boundaries (no previous or next phoneme)
            if i == 0:  # First phoneme in utterance
                continue
            if i == len(phones_tier) - 1:  # Last phoneme in utterance
                continue
            
            # Get phoneme time boundaries
            phoneme_start = phones_tier[i]['xmin']
            phoneme_end = phones_tier[i]['xmax']
            
            # Find corresponding word
            word_info = find_word_at_time(words_tier, (phoneme_start + phoneme_end) / 2)
            word_text = word_info['text'] if word_info else 'unknown'
            word_start = word_info['xmin'] if word_info else phoneme_start
            word_end = word_info['xmax'] if word_info else phoneme_end
            
            # Extract all feature values for different time regions
            # 1. Word time region
            word_features = extract_feature_values(npy_data, word_start, word_end)
            
            # 2. Target phoneme time region
            target_phoneme_features = extract_feature_values(npy_data, phoneme_start, phoneme_end)
            
            # 3. Get context phonemes (previous and next)
            prev_phoneme = phones_tier[i-1]['text'].strip().lower()
            next_phoneme = phones_tier[i+1]['text'].strip().lower()
            
            context_phonemes = [prev_phoneme, current_phoneme, next_phoneme]
            
            # Create phoneme combination name for identification
            combo_name = f"{context_phonemes[0]}_{context_phonemes[1]}_{context_phonemes[2]}"
            
            # Create result entry
            result = {
                'wav_filename': os.path.basename(wav_path),
                'base_name': base_name,
                'phoneme_combination': combo_name,
                'phoneme_xmin': phoneme_start,
                'phoneme_xmax': phoneme_end,
                'word': word_text,
                'word_xmin': word_start,
                'word_xmax': word_end,
                # Word feature values (all frames)
                'word_feature_values': word_features,
                # Target phoneme feature values (all frames)
                'phoneme_combo_feature_values': target_phoneme_features,
                # Individual phoneme features (just the target phoneme in this case)
                'individual_phoneme_features': [target_phoneme_features],
                # Middle phoneme average values
                'middle_phoneme_avg': calculate_feature_averages(target_phoneme_features),
                # Target phonemes list
                'target_phonemes': [target_phoneme]
            }
            
            results.append(result)
    
    return results

def find_word_at_time(words_tier, time_point):
    """
    Find the word that contains the given time point.
    
    Args:
        words_tier: Words tier from TextGrid
        time_point: Time point to search for
    
    Returns:
        dict: Word information or None if not found
    """
    for interval in words_tier:
        if interval['xmin'] <= time_point <= interval['xmax']:
            return {
                'text': interval['text'],
                'xmin': interval['xmin'],
                'xmax': interval['xmax']
            }
    return None

def main():
    parser = argparse.ArgumentParser(description='Process middle phonemes from TextGrid and npy files')
    parser.add_argument('folder_a', help='Path to folder containing npy files')
    parser.add_argument('folder_b', help='Path to folder containing TextGrid and wav files')
    parser.add_argument('phonemes', nargs='+', help='Phonemes to search for as middle phonemes (e.g., ʃ s t͡ʃ)')
    
    args = parser.parse_args()
    
    folder_a = args.folder_a
    folder_b = args.folder_b
    target_phonemes = args.phonemes
    
    print(f"Folder A (npy files): {folder_a}")
    print(f"Folder B (TextGrid/wav files): {folder_b}")
    print(f"Target phonemes: {target_phonemes}")
    
    # Process data
    df = process_middle_phonemes(folder_a, folder_b, target_phonemes)
    
    # Save to CSV (will need to handle the nested data appropriately)
    if not df.empty:
        # For CSV export, we might want to flatten some of the nested data
        flattened_data = []
        for _, row in df.iterrows():
            # Create a flattened version for each feature
            for feature_name in ["LA", "LP", "TBCL", "TBCD", "TTCL", "TTCD"]:
                flattened_row = {
                    'wav_filename': row['wav_filename'],
                    'base_name': row['base_name'],
                    'phoneme_combination': row['phoneme_combination'],
                    'phoneme_xmin': row['phoneme_xmin'],
                    'phoneme_xmax': row['phoneme_xmax'],
                    'word': row['word'],
                    'word_xmin': row['word_xmin'],
                    'word_xmax': row['word_xmax'],
                    'feature_name': feature_name,
                    # Word feature values
                    'word_feature_values': str(row['word_feature_values'][feature_name]),
                    # Phoneme combo feature values
                    'phoneme_combo_feature_values': str(row['phoneme_combo_feature_values'][feature_name]),
                    # Middle phoneme average
                    'middle_phoneme_avg': row['middle_phoneme_avg'].get(feature_name, np.nan)
                }
                flattened_data.append(flattened_row)
        
        flattened_df = pd.DataFrame(flattened_data)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output", "csv")
        os.makedirs(output_dir, exist_ok=True) 
        
        output_csv = os.path.join(output_dir, "phoneme_features.csv")
        
        flattened_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        print(f"Total rows: {len(flattened_df)}")
        
        # Display sample data
        print("\nSample data:")
        print(flattened_df.head(10))
    else:
        print("No data found for the specified phonemes")

if __name__ == "__main__":
    main()