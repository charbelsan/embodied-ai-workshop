# Agent help for the expert challenge

The challenge does not need an agent. If you use one, give it this text:

> Help me improve a drawer-opening policy and integrate it into the mission. Read EXPERT_CHALLENGE.md and notebooks/02_defi_expert_v2.ipynb. You may edit my notebook and my experiments, train and select a model. Propose a hypothesis and compare the reference with my own model using ./expert.sh evaluate. Leave the evaluator and the simulator unchanged, never fabricate a report, and only announce a success from a complete report. Do not act on the robot while a measurement is running. For a custom perception strategy, use only /api/observe_allowed as input; ground truth may label training data, but must not be an input of that strategy. Show the success of the learned opening and the success of the mission separately, with the limits of the sample.

The mission result is measured physically; free changes of strategy are presented at the debrief.
