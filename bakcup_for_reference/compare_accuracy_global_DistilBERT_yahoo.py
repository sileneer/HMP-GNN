# compare_accuracy_global_DistilBERT_yahoo.py
# Compare global accuracy for DistilBERT on Yahoo_Answers_Dataset
# Maintains the same visualization style as compare_accuracy_global_DistilBERT.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Set style for IEEE publication-quality figures
# Use clean, minimal style without heavy grid
plt.style.use('default')

# IEEE-style parameters: clean, professional, publication-ready
plt.rcParams['figure.figsize'] = (6.5, 5)  # IEEE column width (6.5 inches)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif font family
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']  # Arial as primary font
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.borderpad'] = 0.4
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 5

# IEEE-style color palette: professional, distinct colors
# Optimized for maximum distinguishability
IEEE_COLORS = {
    'benign': [
        '#0066CC',  # Blue (Agent 1)
        '#FF6600',  # Orange (Agent 2) 
        '#00B050',  # Green (Agent 3)
        '#FFC000',  # Amber/Yellow (Agent 4)
        '#7030A0',  # Purple (Agent 5)
        '#C55A11',  # Brown (Agent 6)
        '#70AD47',  # Light Green (Agent 7)
        '#5B9BD5',  # Light Blue (Agent 8)
        '#2E75B6',  # Dark Blue (Agent 9)
        '#0070C0',  # Cyan Blue (Agent 10)
        '#954F72',  # Rose (Agent 11)
        '#1F4E79',  # Navy (Agent 12)
        '#000000',  # Black (Agent 13)
        '#C00000',  # Red (Agent 14)
        '#FF0000'   # Bright Red (Agent 15)
    ],
    'attacker': [
        '#DC143C',  # Crimson (Attacker 1)
        '#C00000',  # Dark Red (Attacker 2)
        '#FF4500',  # Orange Red (Attacker 3)
        '#B22222',  # Fire Brick (Attacker 4)
        '#E74C3C',  # Red (Attacker 5)
        '#C0392B',  # Dark Red (Attacker 6)
        '#8B0000',  # Dark Red (Attacker 7)
        '#A52A2A'   # Brown Red (Attacker 8)
    ],
    'global': '#0066CC'  # Professional blue for global accuracy
}

# IEEE-style markers: distinct, professional, optimized for clarity
IEEE_MARKERS = {
    'benign': ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd', '<', '>', 'P', 'H', '8'],
    'attacker': ['s', 'D', '^', 'v', 'p', '*', 'h', 'X']
}

# Comparison plot: explicit colors and markers (GRMP attack in red, others distinct)
# Order: [Without attack, ALIE attack, RMP attack, GRMP attack]
COMPARE_PLOT_COLORS = [
    '#0066CC',   # Blue - Without attack
    '#FF6600',   # Orange - ALIE attack
    '#00B050',   # Green - RMP attack
    '#C00000',   # Red - GRMP attack (proposed method)
]
COMPARE_PLOT_MARKERS = ['o', 's', '^', 'D']  # circle, square, triangle, diamond


def parse_accuracy_data(data_text: str) -> Tuple[List[int], List[float], Dict[str, float]]:
    """
    Parse accuracy data from text format.
    
    Args:
        data_text: Text containing the accuracy table and summary
        
    Returns:
        Tuple of (rounds, accuracies, summary_dict)
        summary_dict contains: 'initial', 'final', 'best', 'change'
    """
    rounds = []
    accuracies = []
    summary = {}
    
    lines = data_text.strip().split('\n')
    
    # Parse the table data
    in_table = False
    for line in lines:
        line = line.strip()
        
        # Skip header lines
        if 'GLOBAL ACCURACY' in line or 'Round' in line or '---' in line or not line:
            if 'Round' in line:
                in_table = True
            continue
        
        # Parse summary line
        if line.startswith('Summary:'):
            # Extract summary values: Initial=0.825333, Final=0.917333, Best=0.921000, Change=+0.092000
            summary_match = re.search(r'Initial=([\d.]+)', line)
            if summary_match:
                summary['initial'] = float(summary_match.group(1))
            
            summary_match = re.search(r'Final=([\d.]+)', line)
            if summary_match:
                summary['final'] = float(summary_match.group(1))
            
            summary_match = re.search(r'Best=([\d.]+)', line)
            if summary_match:
                summary['best'] = float(summary_match.group(1))
            
            summary_match = re.search(r'Change=([\+\-][\d.]+)', line)
            if summary_match:
                summary['change'] = float(summary_match.group(1))
            
            break
        
        # Parse table rows
        if in_table:
            # Format: "1        | 0.825333        |         +0.000000"
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                try:
                    round_num = int(parts[0])
                    accuracy = float(parts[1])
                    rounds.append(round_num)
                    accuracies.append(accuracy)
                except ValueError:
                    continue
    
    return rounds, accuracies, summary


def plot_compare_global_accuracy(
    data_groups: List[Dict],
    save_path: Optional[str] = None,
    results_dir: Path = Path("results")
):
    """
    Plot comparison of global accuracy across different experimental conditions.
    
    Args:
        data_groups: List of dictionaries, each containing:
            - 'name': Label for this experimental condition
            - 'rounds': List of round numbers
            - 'accuracies': List of accuracy values (0-1 scale)
            - 'summary': Optional dict with 'initial', 'final', 'best', 'change'
        save_path: Path to save the figure
        results_dir: Directory for results (if save_path not provided)
    """
    if not data_groups:
        print("  ⚠️  Warning: No data groups provided")
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # IEEE-style: clean, professional appearance
    ax.set_xlabel('Episodes', fontsize=11, fontweight='normal')
    ax.set_ylabel('Global Testing Accuracy (%)', fontsize=11, fontweight='normal')
    
    # Collect all accuracy values for adaptive y-axis range
    all_acc_values = []
    
    # Plot each data group with explicit colors and markers (GRMP in red)
    for i, group in enumerate(data_groups):
        rounds = group['rounds']
        accuracies = group['accuracies']
        name = group.get('name', f'Condition {i+1}')
        
        # Ensure rounds and accuracies have the same length
        min_len = min(len(rounds), len(accuracies))
        if min_len == 0:
            print(f"  ⚠️  Warning: {name} - No data to plot")
            continue
        
        rounds = rounds[:min_len]
        accuracies = accuracies[:min_len]
        
        # Convert to percentage for IEEE style
        accuracies_pct = [acc * 100 for acc in accuracies]
        all_acc_values.extend(accuracies_pct)
        
        # Explicit color and marker: GRMP attack in red with diamond, others from palette
        color = COMPARE_PLOT_COLORS[i % len(COMPARE_PLOT_COLORS)]
        marker = COMPARE_PLOT_MARKERS[i % len(COMPARE_PLOT_MARKERS)]
        
        # Plot accuracy line - solid line, distinct marker
        ax.plot(rounds, accuracies_pct, '-', color=color, 
                linewidth=2, marker=marker, markersize=5, 
                markevery=2,
                label=name, zorder=3, 
                markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=0.5)
    
    # IEEE-style: subtle grid, clean axes
    # Reduce y-axis range to minimize blank space at top
    if all_acc_values:
        data_min = min(all_acc_values)
        data_max = max(all_acc_values)
        data_range = data_max - data_min
        
        # Use smaller padding to minimize blank space
        y_min = max(40, data_min - 2)
        y_max = min(100.0, data_max + 2)  # Small padding to avoid data points at edge
        
        # If range is too small, expand slightly but keep it tight
        if y_max - y_min < 10:
            # Use actual data range with small padding
            padding = max(2, data_range * 0.05)  # 5% padding or at least 2
            y_min = max(0.0, data_min - padding)
            y_max = min(100.0, data_max + padding)
    else:
        y_min, y_max = 0.0, 100.0
    
    # Get max rounds from all groups
    max_round = 1
    for group in data_groups:
        if group['rounds']:
            max_round = max(max_round, max(group['rounds']))
    
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([1, max_round])
    # Background grid: alpha=transparency (0=invisible, 1=opaque)
    ax.grid(True, alpha=0.36, linestyle='--', linewidth=0.5)
    # Show all four borders (full frame)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # IEEE-style legend: clear, professional, positioned at lower left to avoid blocking data
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False,
             edgecolor='black', framealpha=1.0, fontsize=11)
    
    # No title for IEEE style (usually added in LaTeX)
    plt.tight_layout()
    
    out_path = save_path or (results_dir / 'compare_accuracy_global_DistilBERT_yahoo.png')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"  ✅ Saved comparison figure to: {out_path}")
    plt.close()


def main():
    """
    Main function to create comparison plot.
    Add your data groups here.
    """
    # Prepare results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # ============================================================================
    # DATA GROUP 1: First experimental condition
    # ============================================================================
    data_group_1_text = """
--------------------------------------------------------------------------------
1️⃣  GLOBAL ACCURACY (Per Round)
--------------------------------------------------------------------------------
Round    | Clean Accuracy  | Accuracy Change  
--------------------------------------------------------------------------------
1        | 0.526667        |         +0.000000
2        | 0.694000        |         +0.167333
3        | 0.704667        |         +0.010667
4        | 0.708667        |         +0.004000
5        | 0.713000        |         +0.004333
6        | 0.718333        |         +0.005333
7        | 0.718667        |         +0.000333
8        | 0.717333        |         -0.001333
9        | 0.718000        |         +0.000667
10       | 0.719667        |         +0.001667
11       | 0.720000        |         +0.000333
12       | 0.720667        |         +0.000667
13       | 0.722667        |         +0.002000
14       | 0.721000        |         -0.001667
15       | 0.723667        |         +0.002667
16       | 0.724667        |         +0.001000
17       | 0.724333        |         -0.000333
18       | 0.720000        |         -0.004333
19       | 0.721333        |         +0.001333
20       | 0.722000        |         +0.000667
21       | 0.722667        |         +0.000667
22       | 0.720000        |         -0.002667
23       | 0.720333        |         +0.000333
24       | 0.716667        |         -0.003667
25       | 0.719333        |         +0.002667
26       | 0.712000        |         -0.007333
27       | 0.713667        |         +0.001667
28       | 0.714333        |         +0.000667
29       | 0.710000        |         -0.004333
30       | 0.711000        |         +0.001000
31       | 0.709333        |         -0.001667
32       | 0.710333        |         +0.001000
33       | 0.706333        |         -0.004000
34       | 0.708667        |         +0.002333
35       | 0.707667        |         -0.001000
36       | 0.704333        |         -0.003333
37       | 0.702333        |         -0.002000
38       | 0.704667        |         +0.002333
39       | 0.708333        |         +0.003667
40       | 0.701667        |         -0.006667
41       | 0.703000        |         +0.001333
42       | 0.705333        |         +0.002333
43       | 0.704333        |         -0.001000
44       | 0.702333        |         -0.002000
45       | 0.700333        |         -0.002000
46       | 0.700667        |         +0.000333
47       | 0.700667        |         +0.000000
48       | 0.698667        |         -0.002000
49       | 0.695667        |         -0.003000
50       | 0.694667        |         -0.001000
--------------------------------------------------------------------------------
Summary: Initial=0.526667, Final=0.694667, Best=0.724667, Change=+0.168000
"""
    
    # ============================================================================
    # DATA GROUP 2: Second experimental condition
    # ============================================================================
    data_group_2_text = """
--------------------------------------------------------------------------------
1️⃣  GLOBAL ACCURACY (Per Round)
--------------------------------------------------------------------------------
Round    | Clean Accuracy  | Accuracy Change  
--------------------------------------------------------------------------------
1        | 0.276333        |         +0.000000
2        | 0.532667        |         +0.256333
3        | 0.597333        |         +0.064667
4        | 0.616333        |         +0.019000
5        | 0.641000        |         +0.024667
6        | 0.655333        |         +0.014333
7        | 0.663333        |         +0.008000
8        | 0.674667        |         +0.011333
9        | 0.675000        |         +0.000333
10       | 0.682000        |         +0.007000
11       | 0.680667        |         -0.001333
12       | 0.680667        |         +0.000000
13       | 0.688333        |         +0.007667
14       | 0.686333        |         -0.002000
15       | 0.686667        |         +0.000333
16       | 0.686667        |         +0.000000
17       | 0.691333        |         +0.004667
18       | 0.691000        |         -0.000333
19       | 0.694667        |         +0.003667
20       | 0.693667        |         -0.001000
21       | 0.693667        |         +0.000000
22       | 0.694667        |         +0.001000
23       | 0.696000        |         +0.001333
24       | 0.696000        |         +0.000000
25       | 0.698000        |         +0.002000
26       | 0.694667        |         -0.003333
27       | 0.697667        |         +0.003000
28       | 0.693000        |         -0.004667
29       | 0.696333        |         +0.003333
30       | 0.699000        |         +0.002667
31       | 0.695000        |         -0.004000
32       | 0.694667        |         -0.000333
33       | 0.696667        |         +0.002000
34       | 0.693000        |         -0.003667
35       | 0.700333        |         +0.007333
36       | 0.696333        |         -0.004000
37       | 0.693333        |         -0.003000
38       | 0.700000        |         +0.006667
39       | 0.697667        |         -0.002333
40       | 0.697000        |         -0.000667
41       | 0.696333        |         -0.000667
42       | 0.700333        |         +0.004000
43       | 0.698333        |         -0.002000
44       | 0.695667        |         -0.002667
45       | 0.699667        |         +0.004000
46       | 0.698333        |         -0.001333
47       | 0.697333        |         -0.001000
48       | 0.699333        |         +0.002000
49       | 0.699333        |         +0.000000
50       | 0.699000        |         -0.000333
--------------------------------------------------------------------------------
Summary: Initial=0.276333, Final=0.699000, Best=0.700333, Change=+0.422667
"""
    
    # ============================================================================
    # DATA GROUP 3: RMP attack
    # ============================================================================
    data_group_3_text = """
--------------------------------------------------------------------------------
1️⃣  GLOBAL ACCURACY (Per Round)
--------------------------------------------------------------------------------
Round    | Clean Accuracy  | Accuracy Change  
--------------------------------------------------------------------------------
1        | 0.280333        |         +0.000000
2        | 0.536667        |         +0.256333
3        | 0.598000        |         +0.061333
4        | 0.617667        |         +0.019667
5        | 0.643000        |         +0.025333
6        | 0.654000        |         +0.011000
7        | 0.664000        |         +0.010000
8        | 0.673333        |         +0.009333
9        | 0.679333        |         +0.006000
10       | 0.682000        |         +0.002667
11       | 0.679000        |         -0.003000
12       | 0.680333        |         +0.001333
13       | 0.687000        |         +0.006667
14       | 0.685667        |         -0.001333
15       | 0.685667        |         +0.000000
16       | 0.684333        |         -0.001333
17       | 0.689333        |         +0.005000
18       | 0.690667        |         +0.001333
19       | 0.692000        |         +0.001333
20       | 0.693333        |         +0.001333
21       | 0.690333        |         -0.003000
22       | 0.693667        |         +0.003333
23       | 0.695333        |         +0.001667
24       | 0.695667        |         +0.000333
25       | 0.696333        |         +0.000667
26       | 0.696000        |         -0.000333
27       | 0.698000        |         +0.002000
28       | 0.693667        |         -0.004333
29       | 0.698000        |         +0.004333
30       | 0.697333        |         -0.000667
31       | 0.695667        |         -0.001667
32       | 0.698000        |         +0.002333
33       | 0.696000        |         -0.002000
34       | 0.694667        |         -0.001333
35       | 0.696000        |         +0.001333
36       | 0.695333        |         -0.000667
37       | 0.692000        |         -0.003333
38       | 0.701000        |         +0.009000
39       | 0.698667        |         -0.002333
40       | 0.698000        |         -0.000667
41       | 0.700667        |         +0.002667
42       | 0.700000        |         -0.000667
43       | 0.696667        |         -0.003333
44       | 0.697333        |         +0.000667
45       | 0.698333        |         +0.001000
46       | 0.694667        |         -0.003667
47       | 0.698000        |         +0.003333
48       | 0.700667        |         +0.002667
49       | 0.699333        |         -0.001333
50       | 0.700333        |         +0.001000
--------------------------------------------------------------------------------
Summary: Initial=0.280333, Final=0.700333, Best=0.701000, Change=+0.420000
"""
    
    # ============================================================================
    # DATA GROUP 4: GRMP attack
    # ============================================================================
    data_group_4_text = """
--------------------------------------------------------------------------------
1️⃣  GLOBAL ACCURACY (Per Round)
--------------------------------------------------------------------------------
Round    | Clean Accuracy  | Accuracy Change  
--------------------------------------------------------------------------------
1        | 0.462667        |         +0.000000
2        | 0.485667        |         +0.023000
3        | 0.466000        |         -0.019667
4        | 0.437000        |         -0.029000
5        | 0.518667        |         +0.081667
6        | 0.615333        |         +0.096667
7        | 0.596333        |         -0.019000
8        | 0.617000        |         +0.020667
9        | 0.610333        |         -0.006667
10       | 0.652000        |         +0.041667
11       | 0.608667        |         -0.043333
12       | 0.636000        |         +0.027333
13       | 0.640000        |         +0.004000
14       | 0.634000        |         -0.006000
15       | 0.639000        |         +0.005000
16       | 0.637333        |         -0.001667
17       | 0.633333        |         -0.004000
18       | 0.624000        |         -0.009333
19       | 0.641333        |         +0.017333
20       | 0.627667        |         -0.013667
21       | 0.644667        |         +0.017000
22       | 0.631000        |         -0.013667
23       | 0.639667        |         +0.008667
24       | 0.649667        |         +0.010000
25       | 0.642667        |         -0.007000
26       | 0.643667        |         +0.001000
27       | 0.650000        |         +0.006333
28       | 0.647333        |         -0.002667
29       | 0.644667        |         -0.002667
30       | 0.645667        |         +0.001000
31       | 0.646667        |         +0.001000
32       | 0.647667        |         +0.001000
33       | 0.645000        |         -0.002667
34       | 0.649333        |         +0.004333
35       | 0.646667        |         -0.002667
36       | 0.644333        |         -0.002333
37       | 0.643667        |         -0.000667
38       | 0.649000        |         +0.005333
39       | 0.648667        |         -0.000333
40       | 0.650667        |         +0.002000
41       | 0.652333        |         +0.001667
42       | 0.646333        |         -0.006000
43       | 0.650667        |         +0.004333
44       | 0.646667        |         -0.004000
45       | 0.648000        |         +0.001333
46       | 0.656000        |         +0.008000
47       | 0.643000        |         -0.013000
48       | 0.657333        |         +0.014333
49       | 0.654000        |         -0.003333
50       | 0.638000        |         -0.016000
--------------------------------------------------------------------------------
Summary: Initial=0.462667, Final=0.638000, Best=0.657333, Change=+0.175333
"""
    
    # Parse all data groups
    rounds_1, accuracies_1, summary_1 = parse_accuracy_data(data_group_1_text)
    rounds_2, accuracies_2, summary_2 = parse_accuracy_data(data_group_2_text)
    rounds_3, accuracies_3, summary_3 = parse_accuracy_data(data_group_3_text)
    rounds_4, accuracies_4, summary_4 = parse_accuracy_data(data_group_4_text)
    
    # Prepare data groups list
    data_groups = [
        {
            'name': 'Without attack',  # Base model without attack
            'rounds': rounds_1,
            'accuracies': accuracies_1,
            'summary': summary_1
        },
        {
            'name': 'ALIE attack',  # ALIE attack
            'rounds': rounds_2,
            'accuracies': accuracies_2,
            'summary': summary_2
        },
        {
            'name': 'RMP attack',  # RMP attack condition
            'rounds': rounds_3,
            'accuracies': accuracies_3,
            'summary': summary_3
        },
        {
            'name': 'GRMP attack',  # Proposed GRMP attack condition
            'rounds': rounds_4,
            'accuracies': accuracies_4,
            'summary': summary_4
        },
    ]
    
    # Generate comparison plot
    print("\n" + "=" * 60)
    print("Generating Global Accuracy Comparison (DistilBERT, Yahoo_Answers_Dataset)")
    print("=" * 60)
    plot_compare_global_accuracy(data_groups, results_dir=results_dir)
    print("\n✅ DistilBERT Yahoo comparison plot generated successfully!")


if __name__ == "__main__":
    main()
