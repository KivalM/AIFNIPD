import axelrod as axl

# Tournament pools
static_pool = [strategies() for strategies in axl.basic_strategies]
learn_pool = [
    axl.APavlov2006(),
    axl.APavlov2011(),
    axl.AdaptiveTitForTat(),
    axl.AdaptorBrief(),
    axl.AdaptorLong(),
    axl.Adaptive(),
    axl.Calculator(),
    axl.Prober4(),
    axl.RiskyQLearner(),
    axl.CautiousQLearner(),
]

# Result utilities
from .result_utils import (
    get_noise_levels,
    get_agent_dirs,
    load_tournament_results,
    calculate_agent_score,
    calculate_cc_rate,
    calculate_cd_rate,
    load_all_agents_scores,
    load_all_agents_cc_rates,
    load_all_agents_cd_rates,
    get_agent_mean_scores,
    get_all_hyperparams,
    analyze_best_agents_per_noise,
    get_top_n_agents,
    format_results_for_display,
)

# Plotting utilities
from .plot_utils import (
    setup_publication_style,
    get_default_colors_and_markers,
    filter_data_by_agents,
    plot_metric_vs_noise,
    plot_scores_vs_noise,
    plot_cc_rate_vs_noise,
    plot_cd_rate_vs_noise,
    create_comparison_plots,
    generate_plots_for_agent_groups,
    create_summary_table,
    create_formatted_table,
    create_ranking_table,
)

# Statistical utilities
from .stats_utils import (
    levene_test,
    bartlett_test,
    f_test_variance,
    compare_variances_pairwise,
    compare_variances_all,
    ttest_independent,
    mann_whitney_u_test,
    cohens_d,
    hedges_g,
    compare_means_pairwise,
    apply_multiple_comparison_correction,
    anova_oneway,
    kruskal_wallis_test,
    load_agent_repetition_scores,
    compare_agents_at_noise_level,
    create_variance_comparison_summary,
)

__all__ = [
    # Tournament pools
    'static_pool',
    'learn_pool',
    # Result utilities
    'get_noise_levels',
    'get_agent_dirs',
    'load_tournament_results',
    'calculate_agent_score',
    'calculate_cc_rate',
    'calculate_cd_rate',
    'load_all_agents_scores',
    'load_all_agents_cc_rates',
    'load_all_agents_cd_rates',
    'get_agent_mean_scores',
    'get_all_hyperparams',
    'analyze_best_agents_per_noise',
    'get_top_n_agents',
    'format_results_for_display',
    # Plotting utilities
    'setup_publication_style',
    'get_default_colors_and_markers',
    'filter_data_by_agents',
    'plot_metric_vs_noise',
    'plot_scores_vs_noise',
    'plot_cc_rate_vs_noise',
    'plot_cd_rate_vs_noise',
    'create_comparison_plots',
    'generate_plots_for_agent_groups',
    'create_summary_table',
    'create_formatted_table',
    'create_ranking_table',
    # Statistical utilities
    'levene_test',
    'bartlett_test',
    'f_test_variance',
    'compare_variances_pairwise',
    'compare_variances_all',
    'ttest_independent',
    'mann_whitney_u_test',
    'cohens_d',
    'hedges_g',
    'compare_means_pairwise',
    'apply_multiple_comparison_correction',
    'anova_oneway',
    'kruskal_wallis_test',
    'load_agent_repetition_scores',
    'compare_agents_at_noise_level',
    'create_variance_comparison_summary',
]