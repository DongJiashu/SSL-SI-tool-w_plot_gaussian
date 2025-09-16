"""
Phoneme Gaussian Model Training and Evaluation System (Middle Phoneme Only)

Usage Examples:
1. Train new models:
   python xx.py --train-csv train_data.csv --phonemes ɪ t͡ʃ s

2. Evaluate samples with pre-trained models:
   python xx.py --load-model path/to/models.pkl --eval-csv new_samples.csv

3. Train and evaluate in one go:
   python xx.py --train-csv train_data.csv --eval-csv new_samples.csv --phonemes ɪ t͡ʃ

4. Custom threshold (default: 0.05):
   python xx.py --load-model models.pkl --eval-csv new_samples.csv --threshold 0.1
   #threshold is more sensitive when smaller

Parameters:
--train-csv: Path to training CSV file
--eval-csv: Path to evaluation CSV file
--phonemes: Individual phonemes to analyze (space separated)
--load-model: Path to pre-trained model file
--threshold: Probability threshold for anomaly detection (default: 0.05)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import argparse
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from PIL import Image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Phoneme Gaussian Model Training and Evaluation (Middle Phoneme Only)')
    parser.add_argument('--train-csv', type=str, help='Path to the training CSV file')
    parser.add_argument('--eval-csv', type=str, help='Path to the evaluation CSV file (optional)')
    parser.add_argument('--phonemes', type=str, nargs='+', 
                       help='Individual phonemes to analyze (required for training)')
    parser.add_argument('--threshold', type=float, default=0.05, 
                       help='Probability threshold for anomaly detection (default: 0.05)')
    parser.add_argument('--load-model', type=str, help='Path to load pre-trained model (optional)')
    return parser.parse_args()

def extract_middle_phoneme(phoneme_combination: str) -> str:
    """Extract the middle phoneme from phoneme_combination string"""
    if pd.isna(phoneme_combination):
        return ""
    parts = str(phoneme_combination).split('_')
    if len(parts) >= 3:
        return parts[1]  # Middle phoneme is the second part in format "phoneme1_phoneme2_phoneme3"
    elif len(parts) == 1:
        return parts[0]  # Single phoneme case
    else:
        return ""  # Invalid format

def classify_feature(feature_name: str) -> Tuple[str, str]:
    """Classify features into categories"""
    feature_name = feature_name.upper()
    if feature_name in ['LA', 'LP']:
        return 'lips', 'lips'
    elif feature_name.startswith('TBCL') or feature_name.startswith('TBCD'):
        return 'tongue_body', 'tongue_body'
    elif feature_name.startswith('TTCL') or feature_name.startswith('TTCD'):
        return 'tongue_tip', 'tongue_tip'
    else:
        return 'unknown', 'unknown'

def prepare_data(df: pd.DataFrame, phonemes: List[str]) -> Dict[str, pd.DataFrame]:
    """Prepare data by filtering and organizing by middle phoneme only"""
    plot_data = {}
    
    # Extract middle phoneme from phoneme_combination column
    df = df.copy()
    df['middle_phoneme'] = df['phoneme_combination'].apply(extract_middle_phoneme)
    
    for phoneme in phonemes:
        # Filter data where middle_phoneme matches the target phoneme
        phoneme_data = df[df['middle_phoneme'] == phoneme].copy()
        
        if phoneme_data.empty:
            print(f"Warning: No data found for phoneme: {phoneme}")
            continue
        
        plot_data[phoneme] = phoneme_data
    
    return plot_data

def train_gaussian_models(plot_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Train Gaussian models for each phoneme and feature category
    Returns: Dictionary of trained models
    """
    models = {}
    categories = ['tongue_body', 'tongue_tip', 'lips']
    category_features = {
        'tongue_body': ['TBCL', 'TBCD'],
        'tongue_tip': ['TTCL', 'TTCD'], 
        'lips': ['LA', 'LP']
    }
    
    for phoneme, phoneme_data in plot_data.items():
        models[phoneme] = {}
        
        for category in categories:
            location_feature = category_features[category][0]
            degree_feature = category_features[category][1]
            
            # Filter data for this category
            category_data = phoneme_data[
                (phoneme_data['feature_name'].str.upper() == location_feature) | 
                (phoneme_data['feature_name'].str.upper() == degree_feature)
            ]
            
            if category_data.empty:
                continue
            
            # Extract location and degree pairs
            location_dict = {}
            degree_dict = {}
            
            for _, row in category_data.iterrows():
                if row['feature_name'].upper() == location_feature:
                    location_dict[row['base_name']] = row['middle_phoneme_avg']
                elif row['feature_name'].upper() == degree_feature:
                    degree_dict[row['base_name']] = row['middle_phoneme_avg']
            
            # Create paired data
            paired_data = []
            for base_name in set(location_dict.keys()) & set(degree_dict.keys()):
                paired_data.append([location_dict[base_name], degree_dict[base_name]])
            
            if len(paired_data) < 2:  # Need at least 2 points for covariance
                print(f"Warning: Insufficient data for {phoneme} {category} ({len(paired_data)} points)")
                continue
            
            paired_data = np.array(paired_data)
            
            # Calculate mean and covariance
            mean = np.mean(paired_data, axis=0)
            cov = np.cov(paired_data.T)
            
            # Handle singular covariance matrices
            if np.linalg.det(cov) == 0:
                cov += np.eye(2) * 1e-6  # Add small regularization
            
            # Create multivariate normal distribution
            try:
                model = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                models[phoneme][category] = {
                    'model': model,
                    'mean': mean,
                    'cov': cov,
                    'n_samples': len(paired_data),
                    'location_feature': location_feature,
                    'degree_feature': degree_feature,
                    'training_data': paired_data  # Store training data for visualization
                }
            except Exception as e:
                print(f"Error creating model for {phoneme} {category}: {e}")
                continue
    
    return models

def evaluate_sample(sample_data: pd.DataFrame, models: Dict[str, Dict[str, Any]], 
                   threshold: float = 0.05) -> Dict[str, Any]:
    """
    Evaluate a single sample against trained Gaussian models
    Returns: Evaluation results with feedback
    """
    results = {
        'overall_score': 0,
        'is_normal': True,
        'category_results': {},
        'feedback': [],
        'evaluation_points': {},
        'phonemes': []
    }
    
    # Extract middle phoneme from phoneme_combination column
    sample_phoneme_combos = sample_data['phoneme_combination'].unique()
    if len(sample_phoneme_combos) > 0:
        actual_phoneme = extract_middle_phoneme(sample_phoneme_combos[0])
    else:
        actual_phoneme = "unknown"
    
    total_prob = 0
    valid_categories = 0
    evaluated_phonemes = []
    
    for phoneme, phoneme_models in models.items():
        # Only evaluate models that match the sample's middle phoneme
        if phoneme != actual_phoneme:
            continue
            
        evaluated_phonemes.append(phoneme)
        for category, model_info in phoneme_models.items():
            location_feature = model_info['location_feature']
            degree_feature = model_info['degree_feature']
            
            # Get sample values for this category
            location_val = None
            degree_val = None
            
            location_row = sample_data[sample_data['feature_name'].str.upper() == location_feature]
            degree_row = sample_data[sample_data['feature_name'].str.upper() == degree_feature]
            
            if not location_row.empty:
                location_val = location_row['middle_phoneme_avg'].iloc[0]
            if not degree_row.empty:
                degree_val = degree_row['middle_phoneme_avg'].iloc[0]
            
            if location_val is not None and degree_val is not None:
                point = np.array([location_val, degree_val])
                try:
                    probability = model_info['model'].pdf(point)
                    
                    # Normalize probability for better threshold comparison
                    # Calculate probability at mean for normalization reference
                    mean_prob = model_info['model'].pdf(model_info['mean'])
                    if mean_prob > 0:
                        normalized_prob = probability / mean_prob
                    else:
                        normalized_prob = probability
                    
                    # Check if point is within normal range
                    is_normal = normalized_prob > threshold
                    
                    results['category_results'][f"{phoneme}_{category}"] = {
                        'probability': float(probability),
                        'normalized_probability': float(normalized_prob),
                        'is_normal': bool(is_normal),
                        'location_value': float(location_val),
                        'degree_value': float(degree_val),
                        'expected_location_mean': float(model_info['mean'][0]),
                        'expected_degree_mean': float(model_info['mean'][1])
                    }
                    
                    # Store evaluation point for visualization
                    results['evaluation_points'][f"{phoneme}_{category}"] = {
                        'point': point.tolist(),
                        'probability': float(probability),
                        'normalized_probability': float(normalized_prob),
                        'is_normal': bool(is_normal)
                    }
                    
                    total_prob += normalized_prob
                    valid_categories += 1
                    
                    if not is_normal:
                        results['is_normal'] = False
                        # Generate feedback
                        feedback = generate_feedback(phoneme, category, location_val, degree_val, 
                                                   model_info['mean'][0], model_info['mean'][1])
                        results['feedback'].append(feedback)
                    else:
                        # Add positive feedback for normal categories
                        results['feedback'].append(f"{phoneme} {category}: Good performance")
                
                except Exception as e:
                    print(f"Error evaluating {phoneme} {category}: {e}")
                    continue
            else:
                # Add error information for missing features
                missing_features = []
                if location_val is None:
                    missing_features.append(location_feature)
                if degree_val is None:
                    missing_features.append(degree_feature)
                results['category_results'][f"{phoneme}_{category}"] = {
                    'error': f"Missing features: {', '.join(missing_features)}"
                }
    
    # Store only the actually evaluated phonemes
    results['phonemes'] = evaluated_phonemes
    
    if valid_categories > 0:
        results['overall_score'] = total_prob / valid_categories
    
    return results

def generate_feedback(phoneme: str, category: str, actual_loc: float, actual_deg: float,
                     expected_loc: float, expected_deg: float) -> str:
    """Generate specific feedback for improvement"""
    loc_diff = actual_loc - expected_loc
    deg_diff = actual_deg - expected_deg
    
    feedback_parts = []
    
    # Location feedback
    if abs(loc_diff) > 0.1:
        direction = "forward" if loc_diff > 0 else "backward"
        feedback_parts.append(f"adjust constriction location {abs(loc_diff):.3f} units {direction}")
    
    # Degree feedback
    if abs(deg_diff) > 0.1:
        action = "increase" if deg_diff < 0 else "decrease"
        feedback_parts.append(f"{action} constriction degree by {abs(deg_diff):.3f} units")
    
    if feedback_parts:
        return f"For {phoneme} {category}: {', '.join(feedback_parts)}"
    else:
        return f"{phoneme} {category}: Good performance"

def visualize_models(models: Dict[str, Dict[str, Any]], output_dir: str, eval_points: Dict[str, Any] = None):
    """Visualize the Gaussian models with confidence ellipses and evaluation points"""
    for phoneme, phoneme_models in models.items():
        for category, model_info in phoneme_models.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Determine appropriate grid range based on training data and evaluation points
            all_x = []
            all_y = []
            
            # Add training data points
            if 'training_data' in model_info and len(model_info['training_data']) > 0:
                all_x.extend(model_info['training_data'][:, 0])
                all_y.extend(model_info['training_data'][:, 1])
            
            # Add evaluation points if provided
            if eval_points:
                eval_key = f"{phoneme}_{category}"
                if eval_key in eval_points:
                    point = eval_points[eval_key]['point']
                    all_x.append(point[0])
                    all_y.append(point[1])
            
            # Add mean point
            all_x.append(model_info['mean'][0])
            all_y.append(model_info['mean'][1])
            
            # Calculate dynamic range with padding
            if all_x and all_y:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Add 20% padding
                x_padding = x_range * 0.2 if x_range > 0 else 0.5
                y_padding = y_range * 0.2 if y_range > 0 else 0.5
                
                x = np.linspace(x_min - x_padding, x_max + x_padding, 100)
                y = np.linspace(y_min - y_padding, y_max + y_padding, 100)
            else:
                # Fallback to default range
                x = np.linspace(-0.5, 0.5, 100)
                y = np.linspace(-0.5, 0.5, 100)
            
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))
            
            # Calculate PDF
            Z = model_info['model'].pdf(pos)
            
            # Plot contour
            contour = ax.contour(X, Y, Z, levels=10, cmap='viridis', alpha=0.7)
            ax.clabel(contour, inline=True, fontsize=8)
            
            # Plot training data points
            if 'training_data' in model_info and len(model_info['training_data']) > 0:
                ax.scatter(model_info['training_data'][:, 0], model_info['training_data'][:, 1],
                          c='blue', s=30, alpha=0.6, label='Training Data')
            
            # Plot mean
            ax.scatter(model_info['mean'][0], model_info['mean'][1], 
                      c='red', s=100, marker='x', label='Mean')
            
            # Plot confidence ellipse (2σ)
            from matplotlib.patches import Ellipse
            cov = model_info['cov']
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=model_info['mean'],
                         width=lambda_[0]*4, height=lambda_[1]*4,
                         angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
                         edgecolor='red', fc='None', lw=2, linestyle='--', label='2σ Confidence')
            ax.add_patch(ell)
            
            # Plot evaluation points if provided
            if eval_points:
                eval_key = f"{phoneme}_{category}"
                if eval_key in eval_points:
                    point_data = eval_points[eval_key]
                    point = point_data['point']
                    color = 'magenta' if point_data['is_normal'] else 'orange'
                    marker = 'D'
                    size = 120
                    label = 'Normal Sample' if point_data['is_normal'] else 'Anomalous Sample'
                    
                    ax.scatter(point[0], point[1], c=color, s=size, marker=marker, 
                              edgecolors='black', linewidth=2, label=label, alpha=0.8)
            
            ax.set_xlabel('Constriction Location')
            ax.set_ylabel('Constriction Degree')
            ax.set_title(f'Gaussian Model: {phoneme} - {category}\n'
                        f'Mean: ({model_info["mean"][0]:.3f}, {model_info["mean"][1]:.3f})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"model_{phoneme}_{category}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

def combine_phoneme_plots(models: Dict[str, Dict[str, Any]], output_dir: str, 
                         eval_results: Dict[str, Any] = None, threshold: float = 0.05):
    """Combine plots for each phoneme and add feedback text below"""
    # Create a directory for combined plots
    combined_dir = os.path.join(output_dir, "combined_plots")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Get all phonemes from models
    phonemes = list(models.keys())
    
    for phoneme in phonemes:
        # Get all category images for this phoneme
        category_images = []
        categories = []
        
        for category in models[phoneme].keys():
            img_path = os.path.join(output_dir, f"model_{phoneme}_{category}.png")
            if os.path.exists(img_path):
                category_images.append(Image.open(img_path))
                categories.append(category)
        
        if not category_images:
            continue
        
        # Calculate combined width and height
        widths, heights = zip(*(img.size for img in category_images))
        total_width = sum(widths)
        max_height = max(heights)
        
        # Create a new image with enough space for text below
        text_height = 800  # Very large space for huge text
        combined_img = Image.new('RGB', (total_width, max_height + text_height), color='white')
        
        # Paste images side by side
        x_offset = 0
        for img in category_images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        # Add feedback text if evaluation results are available
        if eval_results:
            # Find feedback for this phoneme
            feedback_lines = []
            for sample_name, results in eval_results.items():
                if phoneme in results.get('phonemes', []):
                    for feedback in results.get('feedback', []):
                        if feedback.startswith(f"For {phoneme}"):
                            feedback_lines.append(feedback)
            
            # If no specific feedback found, check for good performance
            if not feedback_lines:
                for sample_name, results in eval_results.items():
                    if phoneme in results.get('phonemes', []):
                        for feedback in results.get('feedback', []):
                            if feedback.startswith(f"{phoneme}") and "Good performance" in feedback:
                                feedback_lines.append(feedback)
            
            # Add feedback text to the image
            if feedback_lines:
                from PIL import ImageDraw, ImageFont
                
                draw = ImageDraw.Draw(combined_img)
                
                # Try to load very large bold fonts
                try:
                    # Try Arial Bold first
                    title_font = ImageFont.truetype("arialbd.ttf", 60)  # Very large bold font
                    feedback_font = ImageFont.truetype("arialbd.ttf", 60)  # Very large bold font
                except:
                    try:
                        # Try regular Arial
                        title_font = ImageFont.truetype("arial.ttf", 60)
                        feedback_font = ImageFont.truetype("arial.ttf", 60)
                    except:
                        try:
                            # Try other common bold fonts
                            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
                            feedback_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
                        except:
                            try:
                                title_font = ImageFont.truetype("Helvetica-Bold.ttf", 60)
                                feedback_font = ImageFont.truetype("Helvetica-Bold.ttf", 60)
                            except:
                                # Final fallback - create default font and make it very bold
                                title_font = ImageFont.load_default()
                                feedback_font = ImageFont.load_default()
                
                # Add threshold information with very large bold font
                threshold_text = f"THRESHOLD: {threshold}"
                # Draw multiple times to make it extremely bold
                for i in range(5):  # Draw 5 times for extreme boldness
                    offset = i
                    draw.text((50 + offset, max_height + 50 + offset), threshold_text, fill='black', font=title_font)
                
                # Add feedback text with very large bold font
                y_text = max_height + 150  # Very large spacing
                for line in feedback_lines:
                    # Draw multiple times to extreme boldness
                    for i in range(5):
                        offset = i
                        draw.text((50 + offset, y_text + offset), line, fill='black', font=feedback_font)
                    
                    y_text += 100  # Very large line spacing
                    
                    # Add a thick separator line
                    if y_text < max_height + text_height - 50:
                        for i in range(3):  # Draw thick line
                            draw.line([(50, y_text - 30 + i), (total_width - 50, y_text - 30 + i)], 
                                     fill='darkgray', width=3)
                        y_text += 50
        
        # Save combined image
        combined_path = os.path.join(combined_dir, f"combined_{phoneme}.png")
        combined_img.save(combined_path)
        print(f"Combined plot saved: {combined_path}")

def save_models(models: Dict[str, Dict[str, Any]], output_dir: str):
    """Save trained models to files"""
    import pickle
    model_path = os.path.join(output_dir, 'gaussian_models.pkl')
    
    # Create a serializable version without the model objects
    serializable_models = {}
    for phoneme, phoneme_models in models.items():
        serializable_models[phoneme] = {}
        for category, model_info in phoneme_models.items():
            serializable_models[phoneme][category] = {
                'mean': model_info['mean'].tolist(),
                'cov': model_info['cov'].tolist(),
                'n_samples': model_info['n_samples'],
                'location_feature': model_info['location_feature'],
                'degree_feature': model_info['degree_feature'],
                'training_data': model_info.get('training_data', []).tolist()
            }
    
    # Save parameters
    with open(model_path, 'wb') as f:
        pickle.dump(serializable_models, f)
    
    # Save as JSON for readability
    json_path = os.path.join(output_dir, 'model_parameters.json')
    with open(json_path, 'w') as f:
        json.dump(serializable_models, f, indent=2)
    
    print(f"Models saved to: {model_path}")
    print(f"Parameters saved to: {json_path}")

def load_models(model_path: str) -> Dict[str, Dict[str, Any]]:
    """Load pre-trained models from file"""
    import pickle
    try:
        with open(model_path, 'rb') as f:
            serializable_models = pickle.load(f)
        
        # Reconstruct multivariate normal models
        models = {}
        for phoneme, phoneme_models in serializable_models.items():
            models[phoneme] = {}
            for category, model_info in phoneme_models.items():
                mean = np.array(model_info['mean'])
                cov = np.array(model_info['cov'])
                
                model = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                models[phoneme][category] = {
                    'model': model,
                    'mean': mean,
                    'cov': cov,
                    'n_samples': model_info['n_samples'],
                    'location_feature': model_info['location_feature'],
                    'degree_feature': model_info['degree_feature'],
                    'training_data': np.array(model_info.get('training_data', []))
                }
        
        print(f"Successfully loaded models from: {model_path}")
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def main():
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("output", "gaussian_models", f"models_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    models = None
    
    # Load pre-trained models if specified
    if args.load_model:
        print("Loading pre-trained models...")
        models = load_models(args.load_model)
        if models is None:
            print("Failed to load models. Exiting.")
            return
    
    # Train new models if no pre-trained models provided
    if models is None and args.train_csv:
        if not args.phonemes:
            print("Error: Phonemes are required for training")
            return
            
        try:
            train_df = pd.read_csv(args.train_csv)
            print(f"Loaded training data with {len(train_df)} rows")
        except Exception as e:
            print(f"Error loading training CSV: {e}")
            return
        
        # Prepare data - filter by middle phoneme only
        plot_data = prepare_data(train_df, args.phonemes)
        if not plot_data:
            print("No valid data found for training")
            return
        
        # Train Gaussian models
        print("Training Gaussian models...")
        models = train_gaussian_models(plot_data)
        
        if not models:
            print("No models were trained successfully")
            return
        
        # Save models
        save_models(models, output_dir)
    
    elif models is None:
        print("Error: Either provide training data or pre-trained models")
        return
    
    # Visualize models
    print("Visualizing models...")
    visualize_models(models, output_dir)
    
    # Evaluate new samples if provided
    eval_results = {}
    if args.eval_csv:
        try:
            eval_df = pd.read_csv(args.eval_csv)
            unique_samples = eval_df['base_name'].unique()
            print(f"\nFound {len(unique_samples)} samples for evaluation")
            print(f"Using threshold: {args.threshold}")
            
            all_eval_points = {}
            
            successful_evals = 0
            failed_evals = 0
            
            for base_name in unique_samples:
                try:
                    sample_data = eval_df[eval_df['base_name'] == base_name]
                    if sample_data.empty:
                        print(f"Warning: No data found for sample {base_name}")
                        failed_evals += 1
                        continue
                    
                    print(f"\nEvaluating sample: {base_name}")
                    results = evaluate_sample(sample_data, models, args.threshold)
                    eval_results[base_name] = results
                    
                    # Collect evaluation points for visualization
                    for key, point_data in results['evaluation_points'].items():
                        all_eval_points[key] = point_data
                    
                    print(f"Middle Phoneme: {results['phonemes'][0] if results['phonemes'] else 'None'}")
                    print(f"Overall Score: {results['overall_score']:.4f}")
                    print(f"Normal: {results['is_normal']}")
                    
                    if results['feedback']:
                        print("Feedback:")
                        for feedback in results['feedback']:
                            print(f"  - {feedback}")
                    else:
                        print("No feedback available")
                    
                    # Print any evaluation errors
                    error_count = 0
                    for category, category_result in results['category_results'].items():
                        if 'error' in category_result:
                            print(f"  Error in {category}: {category_result['error']}")
                            error_count += 1
                    
                    if error_count > 0:
                        print(f"  Total errors: {error_count}")
                    
                    successful_evals += 1
                    
                except Exception as e:
                    print(f"Error evaluating sample {base_name}: {e}")
                    failed_evals += 1
                    continue
            
            print(f"\nEvaluation summary:")
            print(f"Successfully evaluated: {successful_evals} samples")
            print(f"Failed evaluations: {failed_evals} samples")
            
            # Re-visualize with evaluation points
            if successful_evals > 0:
                print("Generating visualizations with evaluation points...")
                visualize_models(models, output_dir, all_eval_points)
            
            # Save evaluation results
            eval_path = os.path.join(output_dir, 'evaluation_results.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Evaluation results saved to: {eval_path}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    # Create combined plots with feedback
    print("Creating combined plots with feedback...")
    combine_phoneme_plots(models, output_dir, eval_results, args.threshold)
    
    print(f"\nAll results saved in: {output_dir}")

if __name__ == "__main__":
    main()