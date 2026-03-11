import pandas as pd
import matplotlib.pyplot as plt

summary_df = pd.read_csv('outputs/summary.csv')


def plot_performance(df):
    sqa_data = df[df['dataset'] == 'strategyqa'].iloc[0]
    labels = ['Raw LLM', 'Greedy Filter', 'FALCON (Soft)']
    values = [sqa_data['em_raw'], sqa_data['em_greedy'], sqa_data['em_falcon']]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['#d3d3d3', '#87ceeb', '#000080'], alpha=0.8)
    plt.ylabel('Exact Match Accuracy', fontsize=12)
    plt.title('StrategyQA Accuracy: Impact of Consistency Enforcement', fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.2)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom',
                 fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('performance_comparison.png')


def plot_contradictions():
    before = 1.62
    after = 0.0
    labels = ['Baseline (Raw)', 'FALCON / Greedy']
    values = [before, after]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['#ff6347', '#4682b4'], alpha=0.8)
    plt.ylabel('Avg. Contradictions per Response', fontsize=12)
    plt.title('FALCON: Effectiveness of Contradiction Resolution', fontsize=14, fontweight='bold')
    plt.ylim(0, 2.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('contradiction_reduction.png')


def plot_milp_graph():
    plt.figure(figsize=(6, 6))
    pos = {'x1': (0.2, 0.8), 'x2': (0.8, 0.8), 'x3': (0.5, 0.2)}  # Node positions

    # Draw edges (Contradictions)
    plt.plot([0.2, 0.8], [0.8, 0.8], 'r--', lw=2, label='Contradiction (Edge)')
    # Draw nodes (Claims)
    plt.scatter([0.2, 0.5], [0.8, 0.2], s=2000, c='green', alpha=0.6, label='Selected ($x_i=1$)')
    plt.scatter([0.8], [0.8], s=2000, c='red', alpha=0.3, label='Rejected ($x_i=0$)')

    for node, (px, py) in pos.items():
        plt.text(px, py, node, ha='center', va='center', fontsize=14, fontweight='bold')

    plt.title('MILP Global Optimization Logic', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig('milp_logic_graph.png')


# --- 4. QUALITATIVE FAILURE CASE GRAPHIC ---
# Visualizes a common error mode where refinement is mistaken for contradiction.
def plot_failure_case():
    plt.figure(figsize=(10, 3.5))
    plt.axis('off')
    text = (
        "FAIL MODE: Contextual Disambiguation Error\n\n"
        "Claim 1: 'The Police (band) cannot make arrests.'\n"
        "Claim 2: 'However, police officers can make arrests.'\n\n"
        "NLI Status: CONTRADICTION (flags 'cannot' vs 'can')\n"
        "MILP Result: Prunes Claim 2 to ensure logical consistency,\n"
        "but loses important contextual refinement."
    )
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=1', fc='white', ec='navy', alpha=0.9))
    plt.title('Qualitative Failure Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('failure_analysis.png')


# Generate all graphics
plot_performance(summary_df)
plot_contradictions()
plot_milp_graph()
plot_failure_case()