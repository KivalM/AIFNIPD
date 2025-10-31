# Experiment Analysis Utilities

This module provides comprehensive utilities for analyzing tournament results, creating publication-quality plots, and performing statistical tests.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Tool: genplots](#cli-tool-genplots)
- [Plotting Utilities](#plotting-utilities)
- [Statistical Testing](#statistical-testing)
- [Examples](#examples)

## Installation

The utilities are already installed as part of the `aifnipd` package. To use the CLI tool:

```bash
# Install dependencies (if not already done)
uv sync

# The genplots command is now available
uv run genplots --help
```

## Quick Start

### Using the CLI Tool

The easiest way to generate all plots is using the `genplots` command:

```bash
# Generate all plots for a results directory
uv run genplots results_1000/main

# This will create a plots/ directory with:
# - All score/CC/CD rate plots (both pools if available) in PDF format
# - Summary tables (CSV files)
# - Ranking tables
```

### Using in Python/Jupyter

```python
from pathlib import Path
import numpy as np
from experiments import (
    load_all_agents_scores,
    plot_scores_vs_noise,
    setup_publication_style,
)

# Setup plotting style
setup_publication_style(use_latex=False)

# Load data
results_dir = Path('results_1000/main/static')
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
scores = load_all_agents_scores(results_dir, noise_levels)

# Create plot
fig, ax = plot_scores_vs_noise(
    scores,
    title='Agent Performance',
    save_path=Path('my_plot')  # Saves as my_plot.pdf
)
```

## CLI Tool: genplots

The `genplots` command provides a convenient way to generate all plots from the command line.

### Basic Usage

```bash
# Generate all plots with default settings
uv run genplots results_1000/main

# Disable LaTeX (faster, no LaTeX installation needed)
uv run genplots results_1000/main --no-latex
```

### Filtering Agents

When you have many agents, you can filter to specific ones:

```bash
# Filter to agents 0, 1, 2, 3, 4, 5, 6, 7
uv run genplots results_1000/main --agent-groups "0,1,2,3,4,5,6,7"

# Multiple groups (creates separate plots for each group)
uv run genplots results_1000/main --agent-groups "[[0,2,4],[1,3,5]]"
```

### Renaming Agents

Make your plots more readable by renaming agents:

```bash
uv run genplots results_1000/main \
    --agent-groups "0,1,2,3" \
    --rename-map '{"0":"AIF-R","1":"AIF-C","2":"QL-R","3":"QL-C"}'
```

### Custom Noise Levels

```bash
uv run genplots results_1000/main --noise-levels "0,0.1,0.2,0.3"
```

### Full Example

```bash
# Complete example with all options
uv run genplots results_1000/main \
    --no-latex \
    --agent-groups "[[0,2,4,6],[1,3,5,7]]" \
    --rename-map '{"0":"AIF-R","1":"AIF-C","2":"QL-R","3":"QL-C","4":"BQL-R","5":"BQL-C","6":"AIF-R-N","7":"AIF-C-N"}' \
    --noise-levels "0,0.05,0.1,0.15,0.2,0.25"
```

## Plotting Utilities

### Setup Publication Style

Configure matplotlib for publication-quality plots:

```python
from experiments import setup_publication_style

# With LaTeX (requires LaTeX installation)
setup_publication_style(use_latex=True)

# Without LaTeX (faster, more compatible)
setup_publication_style(use_latex=False, font_size=10)
```

### Filter and Rename Agents

```python
from experiments import filter_data_by_agents

# Filter by agent indices
filtered_data = filter_data_by_agents(
    data,
    agent_indices=[0, 1, 2, 3],
    rename_map={
        '0': 'AIF-R',
        '1': 'AIF-C',
        '2': 'QL-R',
        '3': 'QL-C',
    }
)
```

### Individual Plots

```python
from experiments import (
    plot_scores_vs_noise,
    plot_cc_rate_vs_noise,
    plot_cd_rate_vs_noise,
)

# Plot scores
fig, ax = plot_scores_vs_noise(
    scores_data,
    title='Agent Performance',
    figsize=(10, 6),
    save_path=Path('score_plot'),
    save_formats=['pdf', 'png']
)

# Plot CC rates
fig, ax = plot_cc_rate_vs_noise(
    cc_data,
    title='Cooperation Rate',
    ylim=(0, 1),
    save_path=Path('cc_plot')
)
```

### Comparison Plots (Side-by-Side)

Note: The CLI tool no longer generates combined comparison plots by default. You can still create them manually if needed:

```python
from experiments import create_comparison_plots

# Compare static vs learning pools
fig, (ax1, ax2) = create_comparison_plots(
    static_scores,
    learning_scores,
    metric_col='mean_score',
    ylabel='Mean Score',
    figsize=(14, 6),
    static_title='Static Pool',
    learning_title='Learning Pool',
    save_path=Path('comparison')
)
```

### Multiple Agent Groups

When you have many agents, create separate plots for different groups:

```python
from experiments import generate_plots_for_agent_groups

# Define agent groups
agent_groups = [
    [0, 2, 4],  # Risky agents
    [1, 3, 5],  # Cautious agents
]

# Generate plots for each group
figures = generate_plots_for_agent_groups(
    scores_data,
    agent_groups,
    plot_scores_vs_noise,
    rename_maps=[rename_map] * len(agent_groups),
    base_save_path=Path('score_by_group'),
    title='Score vs Noise'
)
```

### Summary Tables

```python
from experiments import (
    create_summary_table,
    create_formatted_table,
    create_ranking_table,
)

# Comprehensive summary
summary = create_summary_table(
    scores_data,
    cc_data,
    cd_data,
    pool_name='Static'
)
summary.to_csv('summary.csv', index=False)

# Human-readable formatted table
formatted = create_formatted_table(summary)
formatted.to_csv('formatted_summary.csv', index=False)

# Rankings by noise level
rankings = create_ranking_table(scores_data, pool_name='Static')
rankings.to_csv('rankings.csv', index=False)
```

## Statistical Testing

### Variance Comparison Tests

```python
from experiments import (
    compare_variances_pairwise,
    compare_variances_all,
    levene_test,
)

# Load repetition scores for agents
data_dict = {
    'Agent A': np.array([...]),  # scores across repetitions
    'Agent B': np.array([...]),
    'Agent C': np.array([...]),
}

# Omnibus test - are all variances equal?
omnibus = compare_variances_all(data_dict, test='levene', alpha=0.05)
print(f"Equal variances: {not omnibus['significant']}")

# Pairwise comparisons
pairwise = compare_variances_pairwise(data_dict, test='levene', alpha=0.05)
print(pairwise)
```

### Mean Comparison Tests

```python
from experiments import (
    compare_means_pairwise,
    anova_oneway,
    cohens_d,
)

# ANOVA - do means differ?
anova_result = anova_oneway(data_dict)
print(f"ANOVA p-value: {anova_result['p_value']}")

# Pairwise comparisons with effect sizes
pairwise = compare_means_pairwise(
    data_dict,
    test='ttest',  # or 'welch', 'mann_whitney'
    alpha=0.05,
    correction='holm'  # Multiple comparison correction
)
print(pairwise[['group1', 'group2', 'p_value', 'cohens_d', 'significant']])
```

### Compare Agents at Specific Noise Level

```python
from experiments import compare_agents_at_noise_level

# Compare multiple agents at a specific noise level
results = compare_agents_at_noise_level(
    pool_dir=Path('results_1000/main/static'),
    agent_names=['0_DBS', '1_DBS', '2_QLearning'],
    noise_level=0.15,
    test_variance=True,
    test_means=True,
    alpha=0.05
)

# Results include:
# - variance_omnibus: Overall variance test
# - variance_pairwise: Pairwise variance comparisons
# - means_omnibus_anova: ANOVA results
# - means_pairwise_parametric: t-tests with effect sizes
# - means_pairwise_nonparametric: Mann-Whitney U tests
```

### Variance Summary Across Noise Levels

```python
from experiments import create_variance_comparison_summary

# Compare variance across all noise levels
variance_summary = create_variance_comparison_summary(
    pool_dir=Path('results_1000/main/static'),
    agent_names=['0_DBS', '1_DBS', '2_QLearning'],
    noise_levels=noise_levels
)

# Shows: mean, variance, std, CV for each agent at each noise level
print(variance_summary)
```

## Examples

### Example 1: Quick Plot Generation

```python
from pathlib import Path
import numpy as np
from experiments import *

# Setup
setup_publication_style(use_latex=False)
results_dir = Path('results_1000/main/static')
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))

# Load and plot
scores = load_all_agents_scores(results_dir, noise_levels)
plot_scores_vs_noise(scores, save_path=Path('output/scores'))
```

### Example 2: Filtered Plots with Renaming

```python
from experiments import *

# Load data
scores = load_all_agents_scores(results_dir, noise_levels)

# Filter and rename
rename_map = {'0': 'AIF-R', '1': 'AIF-C', '2': 'QL-R', '3': 'QL-C'}
filtered = filter_data_by_agents(scores, [0, 1, 2, 3], rename_map)

# Plot
plot_scores_vs_noise(filtered, title='Main Agents', save_path=Path('main_agents'))
```

### Example 3: Statistical Comparison

```python
from experiments import *

# Compare agents at noise 0.15
results = compare_agents_at_noise_level(
    pool_dir=Path('results_1000/main/static'),
    agent_names=['0_DBS', '1_DBS', '2_QLearning'],
    noise_level=0.15
)

# Print variance test results
print("Variance Test:")
print(results['variance_omnibus'])

# Print mean comparisons with effect sizes
print("\nMean Comparisons:")
print(results['means_pairwise_parametric'][
    ['group1', 'group2', 'mean_diff', 'p_value', 'cohens_d', 'significant']
])
```

### Example 4: Multiple Agent Groups

```python
from experiments import *

# Define groups
agent_groups = [
    [0, 2, 4],  # Risky agents
    [1, 3, 5],  # Cautious agents
]

rename_maps = [{
    '0': 'AIF-R', '2': 'QL-R', '4': 'BQL-R',
    '1': 'AIF-C', '3': 'QL-C', '5': 'BQL-C',
}] * 2

# Generate separate plots
figures = generate_plots_for_agent_groups(
    scores,
    agent_groups,
    plot_scores_vs_noise,
    rename_maps=rename_maps,
    base_save_path=Path('grouped_plots')
)
```

## Module Structure

```
src/experiments/
├── __init__.py          # Main exports
├── result_utils.py      # Data loading and processing
├── plot_utils.py        # Plotting functions (NEW)
├── stats_utils.py       # Statistical tests (NEW)
├── cli.py              # Command-line interface (NEW)
└── README.md           # This file
```

## Available Functions

### Data Loading (`result_utils.py`)
- `load_all_agents_scores()`
- `load_all_agents_cc_rates()`
- `load_all_agents_cd_rates()`
- `load_tournament_results()`
- `get_agent_dirs()`

### Plotting (`plot_utils.py`)
- `setup_publication_style()`
- `filter_data_by_agents()`
- `plot_scores_vs_noise()`
- `plot_cc_rate_vs_noise()`
- `plot_cd_rate_vs_noise()`
- `create_comparison_plots()`
- `generate_plots_for_agent_groups()`
- `create_summary_table()`
- `create_formatted_table()`
- `create_ranking_table()`

### Statistics (`stats_utils.py`)
- **Variance Tests**: `levene_test()`, `bartlett_test()`, `f_test_variance()`
- **Variance Comparisons**: `compare_variances_pairwise()`, `compare_variances_all()`
- **Mean Tests**: `ttest_independent()`, `mann_whitney_u_test()`
- **Mean Comparisons**: `compare_means_pairwise()`, `anova_oneway()`, `kruskal_wallis_test()`
- **Effect Sizes**: `cohens_d()`, `hedges_g()`
- **Utilities**: `compare_agents_at_noise_level()`, `create_variance_comparison_summary()`

## Tips

1. **Use `--no-latex` for faster plots** during development
2. **Filter agents** when you have many to avoid overcrowded plots
3. **Use agent groups** to create multiple focused plots instead of one cluttered plot
4. **Check variance equality** before choosing between t-test and Welch's t-test
5. **Apply multiple comparison corrections** (Bonferroni, Holm) when doing many pairwise tests
6. **Use effect sizes** (Cohen's d, Hedges' g) to understand practical significance

## Contributing

When adding new functions:
1. Add the function to the appropriate module
2. Export it in `__init__.py`
3. Update this README with examples
4. Add to `__all__` list for proper documentation

## Questions?

See the example notebook at `examples/plot_generation_example.ipynb` for more detailed usage examples.

