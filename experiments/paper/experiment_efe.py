"""
Experiment: EFE decomposition for AIF vs TitForTat.

Runs two DiagnosticAIFAgent configs against axl.TitForTat at noise levels
0, 0.1, 0.2 using axelrod.Match directly (not NoiseTournament) so that
per-timestep internal EFE state is preserved and can be inspected.

Config 1: POMDP formulation  (use_noisy_observation_model=True)
Config 2: MDP  formulation   (use_noisy_observation_model=False)

Results are saved as JSON files under results/paper/experiments/efe/.
"""

import json
from pathlib import Path

import axelrod as axl
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from agents.aif.jax.aif import DiagnosticAIFAgent

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.1, 0.2]
REPETITIONS = 20
TURNS = 100
SEED = 42

GAME = axl.Game(r=3, s=0, t=5, p=1)

OUTPUT_DIR = Path("results/paper/experiments/efe")

AGENT_CONFIGS = {
    "pomdp": dict(
        lr_B=1,
        cooperative_preference=False,
        policy_len=3,
        update_interval=10,
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        use_noisy_observation_model=True,
    ),
    "mdp": dict(
        lr_B=1,
        cooperative_preference=False,
        policy_len=3,
        update_interval=5,
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        use_noisy_observation_model=False,
    ),
}

CONFIG_LABELS = {
    "pomdp": "POMDP (noisy obs)",
    "mdp":   "MDP  (perfect obs)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_single_match(agent_config: dict, noise: float, rep: int, seed: int) -> dict:
    agent = DiagnosticAIFAgent(seed=seed, **agent_config)
    tft = axl.TitForTat()

    match = axl.Match(
        players=(agent, tft),
        turns=TURNS,
        noise=noise,
        game=GAME,
        seed=seed,
    )
    match.play()

    interactions = match.result
    cum_aif = 0.0
    cum_tft = 0.0
    timesteps = []
    efe_history = agent.efe_decomposition_history

    for t, (a_act, t_act) in enumerate(interactions):
        payoffs = GAME.score((a_act, t_act))
        cum_aif += payoffs[0]
        cum_tft += payoffs[1]
        timesteps.append({
            "turn": t,
            "aif_action": a_act.name,
            "tft_action": t_act.name,
            "cumulative_score_aif": cum_aif,
            "cumulative_score_tft": cum_tft,
            "efe": efe_history[t] if t < len(efe_history) else None,
        })

    final_scores = match.final_score()
    return {
        "noise": noise,
        "repetition": rep,
        "seed": seed,
        "turns": TURNS,
        "agent_config": agent_config,
        "final_score_aif": final_scores[0],
        "final_score_tft": final_scores[1],
        "timesteps": timesteps,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_efe_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for config_tag, agent_config in AGENT_CONFIGS.items():
        for noise in NOISE_LEVELS:
            noise_key = f"{noise:.1f}".replace(".", "_")
            output_path = OUTPUT_DIR / f"{config_tag}_noise_{noise_key}.json"

            if output_path.exists():
                print(f"[skip] {output_path} already exists.")
                continue

            print(f"\n=== {config_tag}  noise={noise} ===")
            repetition_results = []

            for rep in range(REPETITIONS):
                rep_seed = int(rng.integers(0, 2**31))
                print(f"  rep {rep + 1}/{REPETITIONS}", end="\r", flush=True)
                result = _run_single_match(agent_config, noise=noise, rep=rep, seed=rep_seed)
                repetition_results.append(result)

            print(f"  rep {REPETITIONS}/{REPETITIONS} — done              ")

            payload = {
                "experiment": "efe_decomposition",
                "config_tag": config_tag,
                "noise": noise,
                "repetitions": REPETITIONS,
                "turns": TURNS,
                "game": {"r": 3, "s": 0, "t": 5, "p": 1},
                "agent": "DiagnosticAIFAgent",
                "opponent": "TitForTat",
                "agent_config": agent_config,
                "results": repetition_results,
            }

            with open(output_path, "w") as f:
                json.dump(payload, f, cls=_NumpyEncoder)
            print(f"  saved → {output_path}")

    print("\nAll experiments done.")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _load_results(config_tag: str, noise: float) -> dict:
    noise_key = f"{noise:.1f}".replace(".", "_")
    path = OUTPUT_DIR / f"{config_tag}_noise_{noise_key}.json"
    with open(path) as f:
        return json.load(f)


EFE_KEYS = ("utility", "state_info_gain", "param_info_gain", "total_neg_efe")


def _extract_arrays(result: dict):
    ts = result["timesteps"]
    n = len(ts)
    out = {f"{a}_{k}": np.zeros(n) for a in ("C", "D") for k in EFE_KEYS}
    out["q_pi_C"] = np.zeros(n)
    out["q_pi_D"] = np.zeros(n)

    for i, step in enumerate(ts):
        efe = step["efe"]
        if efe is None:
            continue
        for a in ("C", "D"):
            for k in EFE_KEYS:
                out[f"{a}_{k}"][i] = efe[a][k]
        out["q_pi_C"][i] = efe["q_pi_C"]
        out["q_pi_D"][i] = efe["q_pi_D"]
    return out


def _average_over_reps(data: dict):
    all_arrs = [_extract_arrays(r) for r in data["results"]]
    numeric_keys = [k for k in all_arrs[0] if isinstance(all_arrs[0][k], np.ndarray)]
    mean, se = {}, {}
    for k in numeric_keys:
        stacked = np.stack([a[k] for a in all_arrs])
        mean[k] = stacked.mean(axis=0)
        se[k] = stacked.std(axis=0) / np.sqrt(len(all_arrs))
    return mean, se


# ---------------------------------------------------------------------------
# Plot 1 — Per-config, per-noise landscape (one figure each)
# ---------------------------------------------------------------------------

def plot_efe_components():
    import matplotlib.pyplot as plt

    for config_tag in AGENT_CONFIGS:
        for noise in NOISE_LEVELS:
            data = _load_results(config_tag, noise)
            mean, se = _average_over_reps(data)
            turns = np.arange(len(mean["q_pi_C"]))

            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(
                f"EFE Decomposition — {CONFIG_LABELS[config_tag]} vs TFT  "
                f"(noise = {noise})",
                fontsize=14, fontweight="bold",
            )

            components = [
                ("utility",         "Utility"),
                ("state_info_gain", "State Info Gain"),
                ("param_info_gain", "Param Info Gain"),
            ]
            colors_c = ["#2196F3", "#4CAF50", "#FF9800"]
            colors_d = ["#1565C0", "#2E7D32", "#E65100"]

            ax = axes[0]
            for idx, (key, label) in enumerate(components):
                for action, colors, ls in [("C", colors_c, "-"), ("D", colors_d, "--")]:
                    m = mean[f"{action}_{key}"]
                    s = se[f"{action}_{key}"]
                    ax.plot(turns, m, color=colors[idx],
                            label=f"{action}: {label}", linewidth=1.2, linestyle=ls)
                    ax.fill_between(turns, m - s, m + s,
                                    color=colors[idx], alpha=0.15)
            ax.set_ylabel("Neg-EFE component")
            ax.legend(ncol=3, fontsize=8, loc="upper right")
            ax.set_title("EFE components (C solid, D dashed)")
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

            ax = axes[1]
            for action, color, ls in [("C", "#2196F3", "-"), ("D", "#E53935", "--")]:
                m = mean[f"{action}_total_neg_efe"]
                s = se[f"{action}_total_neg_efe"]
                ax.plot(turns, m, color=color, label=f"{action} total neg-EFE",
                        linewidth=1.4, linestyle=ls)
                ax.fill_between(turns, m - s, m + s, color=color, alpha=0.15)
            ax.set_ylabel("Total neg-EFE")
            ax.legend(fontsize=9)
            ax.set_title("Total neg-EFE per action")
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

            ax = axes[2]
            ax.plot(turns, mean["q_pi_C"], color="#2196F3",
                    label="P(Cooperate)", linewidth=1.4)
            ax.fill_between(turns, mean["q_pi_C"] - se["q_pi_C"],
                            mean["q_pi_C"] + se["q_pi_C"],
                            color="#2196F3", alpha=0.15)
            ax.plot(turns, mean["q_pi_D"], color="#E53935",
                    label="P(Defect)", linewidth=1.4)
            ax.fill_between(turns, mean["q_pi_D"] - se["q_pi_D"],
                            mean["q_pi_D"] + se["q_pi_D"],
                            color="#E53935", alpha=0.15)
            ax.set_ylabel("Action probability")
            ax.set_xlabel("Turn")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=9)
            ax.set_title("Marginal action probabilities")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out = OUTPUT_DIR / f"landscape_{config_tag}_noise_{noise:.1f}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Plot 2 — D-C difference per component, all noise levels on one figure
# ---------------------------------------------------------------------------

def plot_component_difference():
    import matplotlib.pyplot as plt

    components = [
        ("utility",         "Utility"),
        ("state_info_gain", "State Info Gain"),
        ("param_info_gain", "Param Info Gain"),
        ("total_neg_efe",   "Total neg-EFE"),
    ]
    noise_colors = {0.0: "#2196F3", 0.1: "#FF9800", 0.2: "#E53935"}

    for config_tag in AGENT_CONFIGS:
        fig, axes = plt.subplots(len(components), 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"EFE Difference (D - C) — {CONFIG_LABELS[config_tag]}",
            fontsize=14, fontweight="bold",
        )

        for ax, (key, label) in zip(axes, components):
            for noise in NOISE_LEVELS:
                data = _load_results(config_tag, noise)
                mean, se = _average_over_reps(data)
                turns = np.arange(len(mean["q_pi_C"]))
                diff = mean[f"D_{key}"] - mean[f"C_{key}"]
                diff_se = np.sqrt(se[f"D_{key}"]**2 + se[f"C_{key}"]**2)
                ax.plot(turns, diff, color=noise_colors[noise],
                        label=f"noise={noise}", linewidth=1.2)
                ax.fill_between(turns, diff - diff_se, diff + diff_se,
                                color=noise_colors[noise], alpha=0.12)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            ax.set_ylabel("D − C")
            ax.set_title(label)
            if ax is axes[0]:
                ax.legend(fontsize=9)

        axes[-1].set_xlabel("Turn")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = OUTPUT_DIR / f"diff_{config_tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Plot 3 — POMDP vs MDP comparison (the main contrast plot)
# ---------------------------------------------------------------------------

def plot_config_comparison():
    """Side-by-side comparison of both configs at each noise level.

    Each figure (one per noise level) has 4 rows x 2 columns:
      columns = POMDP | MDP
      rows    = Utility(D-C) | StateIG(D-C) | ParamIG(D-C) | P(Cooperate)
    """
    import matplotlib.pyplot as plt

    components = [
        ("utility",         "Utility (D − C)"),
        ("state_info_gain", "State Info Gain (D − C)"),
        ("param_info_gain", "Param Info Gain (D − C)"),
    ]
    config_tags = list(AGENT_CONFIGS.keys())
    config_colors = {"pomdp": "#2196F3", "mdp": "#E53935"}

    for noise in NOISE_LEVELS:
        fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
        fig.suptitle(
            f"POMDP vs MDP — AIF vs TFT  (noise = {noise})",
            fontsize=15, fontweight="bold",
        )

        for col, config_tag in enumerate(config_tags):
            data = _load_results(config_tag, noise)
            mean, se = _average_over_reps(data)
            turns = np.arange(len(mean["q_pi_C"]))
            color = config_colors[config_tag]

            for row, (key, label) in enumerate(components):
                ax = axes[row, col]
                diff = mean[f"D_{key}"] - mean[f"C_{key}"]
                diff_se = np.sqrt(se[f"D_{key}"]**2 + se[f"C_{key}"]**2)
                ax.plot(turns, diff, color=color, linewidth=1.3)
                ax.fill_between(turns, diff - diff_se, diff + diff_se,
                                color=color, alpha=0.15)
                ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
                if col == 0:
                    ax.set_ylabel("D − C")
                if row == 0:
                    ax.set_title(CONFIG_LABELS[config_tag], fontsize=12,
                                 fontweight="bold")
                # Row label on right side
                if col == 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(label, fontsize=10, rotation=270, labelpad=18)

            # Bottom row: P(Cooperate)
            ax = axes[3, col]
            ax.plot(turns, mean["q_pi_C"], color=color, linewidth=1.4,
                    label="P(Cooperate)")
            ax.fill_between(turns, mean["q_pi_C"] - se["q_pi_C"],
                            mean["q_pi_C"] + se["q_pi_C"],
                            color=color, alpha=0.15)
            ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Turn")
            if col == 0:
                ax.set_ylabel("P(Cooperate)")
            if col == 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("P(Cooperate)", fontsize=10,
                              rotation=270, labelpad=18)
            if row == 0:
                ax.set_title(CONFIG_LABELS[config_tag], fontsize=12,
                             fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = OUTPUT_DIR / f"comparison_noise_{noise:.1f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Overlay both configs on same axes for direct contrast
# ---------------------------------------------------------------------------

def plot_config_overlay():
    """One figure per noise level with both configs overlaid.

    4 panels: 3 EFE component differences (D-C) + P(Cooperate).
    Each panel shows both POMDP and MDP as separate lines.
    """
    import matplotlib.pyplot as plt

    components = [
        ("utility",         "Utility (D − C)"),
        ("state_info_gain", "State Info Gain (D − C)"),
        ("param_info_gain", "Param Info Gain (D − C)"),
    ]
    config_colors = {"pomdp": "#2196F3", "mdp": "#E53935"}
    config_ls     = {"pomdp": "-",       "mdp": "--"}

    for noise in NOISE_LEVELS:
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(
            f"POMDP vs MDP overlay — AIF vs TFT  (noise = {noise})",
            fontsize=14, fontweight="bold",
        )

        for config_tag in AGENT_CONFIGS:
            data = _load_results(config_tag, noise)
            mean, se = _average_over_reps(data)
            turns = np.arange(len(mean["q_pi_C"]))
            color = config_colors[config_tag]
            ls = config_ls[config_tag]
            lbl = CONFIG_LABELS[config_tag]

            for row, (key, title) in enumerate(components):
                ax = axes[row]
                diff = mean[f"D_{key}"] - mean[f"C_{key}"]
                diff_se = np.sqrt(se[f"D_{key}"]**2 + se[f"C_{key}"]**2)
                ax.plot(turns, diff, color=color, linestyle=ls, linewidth=1.3,
                        label=lbl)
                ax.fill_between(turns, diff - diff_se, diff + diff_se,
                                color=color, alpha=0.12)

            ax = axes[3]
            ax.plot(turns, mean["q_pi_C"], color=color, linestyle=ls,
                    linewidth=1.4, label=lbl)
            ax.fill_between(turns, mean["q_pi_C"] - se["q_pi_C"],
                            mean["q_pi_C"] + se["q_pi_C"],
                            color=color, alpha=0.12)

        for row, (key, title) in enumerate(components):
            axes[row].axhline(0, color="grey", linewidth=0.5, linestyle=":")
            axes[row].set_ylabel("D − C")
            axes[row].set_title(title)
            axes[row].legend(fontsize=9)

        axes[3].axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_ylabel("P(Cooperate)")
        axes[3].set_title("P(Cooperate)")
        axes[3].set_xlabel("Turn")
        axes[3].legend(fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = OUTPUT_DIR / f"overlay_noise_{noise:.1f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved → {out}")


def plot_all():
    print("\nGenerating plots...")
    plot_efe_components()
    plot_component_difference()
    plot_config_comparison()
    plot_config_overlay()
    print("Done.")


if __name__ == "__main__":
    run_efe_experiment()
    plot_all()
