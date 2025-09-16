"""
Original Author : Yashish Maduwantha
updated by: Jiashu Dong 

Runs the SSL_SI models on WAV files specified in a CSV file

1. Read WAV file paths from CSV
2. Run feature extractor to extract SSL acoustic features
3. Run the pretrained SI system to estimate and save the TVs

Modified to process files from CSV and support parallel processing
"""
import os
import argparse
import csv
import multiprocessing as mp
from tqdm import tqdm
from run_saved_model import run_model
from feature_extracter import feature_extract

def get_parser():
    """
    :Description: Returns a parser with custom arguments cli
    :return: parser
    """
    parser = argparse.ArgumentParser(description='Run the SI pipeline on CSV-specified WAV files',
                                     epilog="Process WAV files from CSV, extract features and generate TVs")
    parser.add_argument('-m', '--model', type=str, default='xrmb',
                        help='set which SI system to run, xrmb trained (xrmb) or hprc trained (hprc)')
    parser.add_argument('-f', '--feats', type=str, default='hubert',
                        help='set which SSL pretrained model to be used to extract features')
    parser.add_argument('-c', '--csv_file', type=str, required=True,
                        help='path to CSV file containing wav_file_path and phone_contexts')
    parser.add_argument('-o', '--out_format', type=str, default='mat',
                        help='output TV file format (mat or npy)')
    parser.add_argument('-j', '--workers', type=int, default=4,
                        help='number of parallel workers (default: 4)')
    return parser

def read_wav_files_from_csv(csv_file):
    """
    Read WAV file paths from CSV file
    
    :param csv_file: Path to CSV file
    :return: List of WAV file paths
    """
    wav_files = []
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row and row[0].endswith('.wav'):
                wav_files.append(row[0])
    
    return wav_files

def process_single_file(args):
    """
    Process a single WAV file (for parallel processing)
    
    :param args: Tuple of (audio_file, feats, SI_model, out_format)
    :return: (success, audio_file, error_message)
    """
    audio_file, feats, SI_model, out_format = args
    
    try:
        # Create feature extractor instance for this process
        f_extractor = feature_extract(feats)
        
        # Get filename without extension
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Run feature extraction
        feature_data, no_segs, audio_len = f_extractor.run_extraction(audio_file)
        
        # Run model to generate TVs
        run_model(feature_data, file_name, audio_len, SI_model, out_format=out_format)
        
        return (True, audio_file, None)
        
    except Exception as e:
        return (False, audio_file, str(e))

def main():
    args = get_parser().parse_args()

    SI_model = args.model
    feats = args.feats
    csv_file = args.csv_file
    out_format = args.out_format
    num_workers = args.workers
    
    # Read WAV files from CSV
    try:
        wav_files = read_wav_files_from_csv(csv_file)
    except FileNotFoundError as e:
        print(e)
        return
    
    if not wav_files:
        print("No WAV files found in the CSV file")
        return
    
    print(f"Found {len(wav_files)} WAV file(s) to process from CSV")
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    process_args = [(file, feats, SI_model, out_format) for file in wav_files]
    
    # Process files in parallel
    success_count = 0
    failed_files = []
    
    if num_workers > 1:
        # Use multiprocessing for parallel processing
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_file, process_args), 
                              total=len(wav_files), 
                              desc="Processing WAV files"))
            
            for success, file, error in results:
                if success:
                    success_count += 1
                else:
                    failed_files.append((file, error))
    else:
        # Single process mode (for debugging)
        for args in tqdm(process_args, desc="Processing WAV files"):
            success, file, error = process_single_file(args)
            if success:
                success_count += 1
            else:
                failed_files.append((file, error))
    
    # Print results
    print(f"\nProcessing completed:")
    print(f"Successfully processed: {success_count}/{len(wav_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file, error in failed_files:
            print(f"  {file}: {error}")

if __name__ == '__main__':
    main()