import axelrod as axl

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