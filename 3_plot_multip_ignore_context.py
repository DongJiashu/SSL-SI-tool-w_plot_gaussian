import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import List, Dict, Tuple
import numpy as np
from matplotlib.colors import to_rgba

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate phoneme comparison plots from CSV data')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('phonemes', type=str, nargs='+', 
                       help='Phonemes to compare (e.g., ɪ i) - can specify multiple')
    return parser.parse_args()

def classify_feature(feature_name: str) -> Tuple[str, str]:
    """
    Classify features into categories
    Returns: (value_type, feature_category)
    """
    feature_name = feature_name.upper()
    if feature_name in ['LA', 'LP']:
        return 'lips', 'lips'
    elif feature_name.startswith('TBCL') or feature_name.startswith('TBCD'):
        return 'tongue_body', 'tongue_body'
    elif feature_name.startswith('TTCL') or feature_name.startswith('TTCD'):
        return 'tongue_tip', 'tongue_tip'
    else:
        return 'unknown', 'unknown'

def prepare_plot_data(df: pd.DataFrame, phonemes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for plotting by filtering and organizing by phonemes
    """
    plot_data = {}
    
    for phoneme in phonemes:
        # Extract the middle phoneme from each combination
        df['middle_phoneme'] = df['phoneme_combination'].apply(
            lambda x: x.split('_')[1] if isinstance(x, str) and len(x.split('_')) >= 3 else None
        )
        
        # Filter by the specified phoneme
        phoneme_data = df[df['middle_phoneme'] == phoneme].copy()
        
        if phoneme_data.empty:
            print(f"Warning: No data found for phoneme: {phoneme}")
            available_phonemes = df['middle_phoneme'].unique()
            print(f"Available phonemes: {available_phonemes}")
            continue
        
        print(f"Found {len(phoneme_data)} rows for phoneme: {phoneme}")
        
        # Store all data points for this phoneme
        plot_data[phoneme] = phoneme_data
    
    return plot_data

def create_combined_plot(plot_data: Dict[str, pd.DataFrame], output_filename: str):
    """
    Create a combined plot with all three categories on the same x-axis
    """
    categories = ['tongue_body', 'tongue_tip', 'lips']
    category_features = {
        'tongue_body': ['TBCL', 'TBCD'],
        'tongue_tip': ['TTCL', 'TTCD'], 
        'lips': ['LA', 'LP']
    }
    
    # Define x-offsets for each category
    category_offsets = {
        'tongue_body': 0,
        'tongue_tip': 1.5,  # Increased spacing between categories
        'lips': 3.0 
    }
    
    # Use a color palette with good distinction
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'd', 'h']
    
    # Set up the combined plot
    fig, ax = plt.subplots(figsize=(18, 10))  # Increased figure size
    
    legend_handles = []
    plotted_phonemes = set()
    total_points = 0
    
    # Add jitter to reduce overplotting
    jitter_amount = 0.02
    
    for i, (phoneme, phoneme_data) in enumerate(plot_data.items()):
        color = colors[i]
        marker = markers[i % len(markers)]
        
        # Plot data for each category
        for category in categories:
            location_feature = category_features[category][0]
            degree_feature = category_features[category][1]
            x_offset = category_offsets[category]
            
            # Filter data for this category
            category_data = phoneme_data[
                (phoneme_data['feature_name'].str.upper() == location_feature) | 
                (phoneme_data['feature_name'].str.upper() == degree_feature)
            ]
            
            if category_data.empty:
                continue
            
            # Separate location and degree values
            location_dict = {}
            degree_dict = {}
            
            for _, row in category_data.iterrows():
                if row['feature_name'].upper() == location_feature:
                    # Use base_name as key to match corresponding values
                    location_dict[row['base_name']] = row['middle_phoneme_avg']
                elif row['feature_name'].upper() == degree_feature:
                    degree_dict[row['base_name']] = row['middle_phoneme_avg']
            
            # Plot matching pairs with jitter to reduce overplotting
            matched_points = []
            for base_name in set(location_dict.keys()) & set(degree_dict.keys()):
                x = location_dict[base_name] + x_offset + np.random.normal(0, jitter_amount)
                y = degree_dict[base_name] + np.random.normal(0, jitter_amount)
                matched_points.append((x, y))
            
            # Plot as a single scatter for better performance and reduced clutter
            if matched_points:
                xs, ys = zip(*matched_points)
                if phoneme not in plotted_phonemes:
                    scatter = ax.scatter(xs, ys, color=color, s=80, alpha=0.7, 
                                       label=f'/{phoneme}/', edgecolors='darkgray', linewidth=0.5,
                                       marker=marker)
                    legend_handles.append(scatter)
                    plotted_phonemes.add(phoneme)
                else:
                    ax.scatter(xs, ys, color=color, s=80, alpha=0.7, 
                             edgecolors='darkgray', linewidth=0.5, marker=marker)
                
                total_points += len(matched_points)
            
            # Plot unmatched location values with reduced opacity
            unmatched_x = []
            for base_name in set(location_dict.keys()) - set(degree_dict.keys()):
                x = location_dict[base_name] + x_offset + np.random.normal(0, jitter_amount)
                unmatched_x.append(x)
            
            if unmatched_x:
                if phoneme not in plotted_phonemes:
                    scatter = ax.scatter(unmatched_x, [0] * len(unmatched_x), 
                                       color=to_rgba(color, 0.4), s=60, 
                                       label=f'/{phoneme}/ (location only)', 
                                       edgecolors='none', marker=marker)
                    legend_handles.append(scatter)
                    plotted_phonemes.add(phoneme)
                else:
                    ax.scatter(unmatched_x, [0] * len(unmatched_x), 
                             color=to_rgba(color, 0.4), s=60, 
                             edgecolors='none', marker=marker)
                
                total_points += len(unmatched_x)
            
            # Plot unmatched degree values with reduced opacity
            unmatched_y = []
            for base_name in set(degree_dict.keys()) - set(location_dict.keys()):
                y = degree_dict[base_name] + np.random.normal(0, jitter_amount)
                unmatched_y.append(y)
            
            if unmatched_y:
                if phoneme not in plotted_phonemes:
                    scatter = ax.scatter([x_offset] * len(unmatched_y), unmatched_y, 
                                       color=to_rgba(color, 0.4), s=60, 
                                       label=f'/{phoneme}/ (degree only)', 
                                       edgecolors='none', marker=marker)
                    legend_handles.append(scatter)
                    plotted_phonemes.add(phoneme)
                else:
                    ax.scatter([x_offset] * len(unmatched_y), unmatched_y, 
                             color=to_rgba(color, 0.4), s=60, 
                             edgecolors='none', marker=marker)
                
                total_points += len(unmatched_y)
    
    if not plotted_phonemes:
        print("No data found for plotting")
        plt.close(fig)
        return
    
    # Set labels and title
    ax.set_xlabel('Vocal Tract Constriction Location', fontsize=14, labelpad=15)
    ax.set_ylabel('Vocal Tract Constriction Degree', fontsize=14)
    
    # Create title
    phoneme_names = [f'/{p}/' for p in plot_data.keys()]
    title = f"Comparison of Phonemes: {', '.join(phoneme_names)}"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    x_ticks = []
    x_tick_labels = []
    
    for category in categories:
        offset = category_offsets[category]
        # Add ticks for this category range
        x_ticks.extend([offset - 0.5, offset - 0.25, offset, offset + 0.25])
        x_tick_labels.extend(['-0.50', '-0.25', '0.00', '0.25'])
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=11)
    
    # Add vertical lines to separate categories
    separator_positions = [category_offsets['tongue_tip'] - 0.75, category_offsets['lips'] - 0.75]
    for sep_x in separator_positions:
        ax.axvline(x=sep_x, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add category labels above the plot
    for category in categories:
        offset = category_offsets[category]
        category_center = offset  # Center of this category's range
        ax.text(category_center, 0.45, category.replace('_', ' ').title(), 
               ha='center', va='center', fontsize=13, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    # Set y-axis range and ticks
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, -0.25, 0.00, 0.25, 0.50])
    ax.tick_params(axis='y', labelsize=11)
    
    # Set x-axis limits to show all categories nicely
    ax.set_xlim(-0.5, category_offsets['lips'] + 0.5)
    
    # Add grid with lighter appearance
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Add legend outside the plot to avoid clutter
    ax.legend(handles=legend_handles, loc='center left', 
              bbox_to_anchor=(1, 0.5), fontsize=11, 
              framealpha=0.9, shadow=True)
    
    # Add total data points info
    ax.text(0.02, 0.98, f'Total data points: {total_points}', 
           transform=ax.transAxes, verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved as: {output_filename}")
    print(f"  - Contains {len(plotted_phonemes)} phonemes")
    print(f"  - Total data points: {total_points}")

def create_individual_plots(plot_data: Dict[str, pd.DataFrame], output_prefix: str):
    """
    Create separate plots for each feature category
    """
    categories = ['tongue_body', 'tongue_tip', 'lips']
    category_features = {
        'tongue_body': ['TBCL', 'TBCD'],
        'tongue_tip': ['TTCL', 'TTCD'], 
        'lips': ['LA', 'LP']
    }
    
    # Use a color palette with good distinction
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'd', 'h']
    
    for category in categories:
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))  # Slightly larger for individual plots
        
        legend_handles = []
        plotted_phonemes = set()
        total_points = 0
        
        location_feature = category_features[category][0]
        degree_feature = category_features[category][1]
        
        # Add jitter to reduce overplotting
        jitter_amount = 0.02
        
        for i, (phoneme, phoneme_data) in enumerate(plot_data.items()):
            color = colors[i]
            marker = markers[i % len(markers)]
            
            # Filter data for this category
            category_data = phoneme_data[
                (phoneme_data['feature_name'].str.upper() == location_feature) | 
                (phoneme_data['feature_name'].str.upper() == degree_feature)
            ]
            
            if category_data.empty:
                print(f"No {category} data found for phoneme: {phoneme}")
                continue
            
            # Separate location and degree values
            location_dict = {}
            degree_dict = {}
            
            for _, row in category_data.iterrows():
                if row['feature_name'].upper() == location_feature:
                    location_dict[row['base_name']] = row['middle_phoneme_avg']
                elif row['feature_name'].upper() == degree_feature:
                    degree_dict[row['base_name']] = row['middle_phoneme_avg']
            
            # Plot matching pairs with jitter
            matched_points = []
            for base_name in set(location_dict.keys()) & set(degree_dict.keys()):
                x = location_dict[base_name] + np.random.normal(0, jitter_amount)
                y = degree_dict[base_name] + np.random.normal(0, jitter_amount)
                matched_points.append((x, y))
            
            if matched_points:
                xs, ys = zip(*matched_points)
                if phoneme not in plotted_phonemes:
                    scatter = ax.scatter(xs, ys, color=color, s=80, alpha=0.7, 
                                       label=f'/{phoneme}/', edgecolors='darkgray', linewidth=0.5,
                                       marker=marker)
                    legend_handles.append(scatter)
                    plotted_phonemes.add(phoneme)
                else:
                    ax.scatter(xs, ys, color=color, s=80, alpha=0.7, 
                             edgecolors='darkgray', linewidth=0.5, marker=marker)
                
                total_points += len(matched_points)
            
            # Plot unmatched values with reduced opacity
            unmatched_x = []
            for base_name in set(location_dict.keys()) - set(degree_dict.keys()):
                x = location_dict[base_name] + np.random.normal(0, jitter_amount)
                unmatched_x.append(x)
            
            if unmatched_x:
                if phoneme not in plotted_phonemes:
                    scatter = ax.scatter(unmatched_x, [0] * len(unmatched_x), 
                                       color=to_rgba(color, 0.4), s=60, 
                                       label=f'/{phoneme}/ (location only)', 
                                       edgecolors='none', marker=marker)
                    legend_handles.append(scatter)
                    plotted_phonemes.add(phoneme)
                else:
                    ax.scatter(unmatched_x, [0] * len(unmatched_x), 
                             color=to_rgba(color, 0.4), s=60, 
                             edgecolors='none', marker=marker)
                
                total_points += len(unmatched_x)
            
            unmatched_y = []
            for base_name in set(degree_dict.keys()) - set(location_dict.keys()):
                y = degree_dict[base_name] + np.random.normal(0, jitter_amount)
                unmatched_y.append(y)
            
            if unmatched_y:
                if phoneme not in plotted_phonemes:
                    scatter = ax.scatter([0] * len(unmatched_y), unmatched_y, 
                                       color=to_rgba(color, 0.4), s=60, 
                                       label=f'/{phoneme}/ (degree only)', 
                                       edgecolors='none', marker=marker)
                    legend_handles.append(scatter)
                    plotted_phonemes.add(phoneme)
                else:
                    ax.scatter([0] * len(unmatched_y), unmatched_y, 
                             color=to_rgba(color, 0.4), s=60, 
                             edgecolors='none', marker=marker)
                
                total_points += len(unmatched_y)
        
        if not plotted_phonemes:
            print(f"No data found for category: {category}")
            plt.close(fig)
            continue
        
        # Set labels and title
        ax.set_xlabel('Vocal Tract Constriction Location', fontsize=12)
        ax.set_ylabel('Vocal Tract Constriction Degree', fontsize=12)
        
        # Create title
        phoneme_names = [f'/{p}/' for p in plot_data.keys()]
        category_title = category.replace('_', ' ').title()
        title = f"Comparison of {category_title}: {', '.join(phoneme_names)}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set axis ranges
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.75, 0.5)
        
        # Set ticks
        ax.set_xticks([-0.5, -0.25, 0.00, 0.25, 0.5])
        ax.set_yticks([-0.75, -0.5, -0.25, 0.00, 0.25, 0.50])
        
        # Add grid with lighter appearance
        ax.grid(True, alpha=0.2)
        
        # Add legend outside the plot
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add data points info
        ax.text(0.02, 0.98, f'Total data points: {total_points}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Adjust layout to accommodate external legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        output_filename = f"{output_prefix}_{category}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plot saved as: {output_filename}")

def main():
    args = parse_arguments()
    
    # Read CSV file
    try:
        df = pd.read_csv(args.csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Available features: {', '.join(df['feature_name'].unique())}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Prepare plot data
    plot_data = prepare_plot_data(df, args.phonemes)
    
    if not plot_data:
        print("No valid data found for the specified phonemes")
        return
    
    # Print data summary
    print("\nData summary:")
    for phoneme, data in plot_data.items():
        print(f"  /{phoneme}/: {len(data)} data points")
        for feature in data['feature_name'].unique():
            count = len(data[data['feature_name'] == feature])
            print(f"    {feature}: {count} points")
    
    # Create output directory with timestamp
    from datetime import datetime
    
    # Get script directory and create output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(script_dir, "output", "plot", f"plots_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory created: {output_dir}")
    
    # Generate output filenames
    phoneme_str = '_'.join(args.phonemes).replace(' ', '_').replace('/', '')
    
    # Create combined plot
    combined_output = os.path.join(output_dir, f"phoneme_comparison_combined_{phoneme_str}.png")
    create_combined_plot(plot_data, combined_output)
    
    # Create individual plots
    individual_prefix = os.path.join(output_dir, f"phoneme_comparison_individual_{phoneme_str}")
    create_individual_plots(plot_data, individual_prefix)
    
    print("\nAll plots generated successfully!")
    print(f"Plots saved in: {output_dir}")

if __name__ == "__main__":
    main()