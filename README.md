# Generic-to-Specific Reasoning and Ad Hoc Teamwork in Embodied AI Agents
This repository contains the supplementary files for a hybrid architecture for ad hoc teamwork combining knowledge based reasoning with data-driven learning, while using Large Language Models (LLM) to predict future tasks of the agents.

## Folder structure

```bash
.
├── ASP/ahagent.sp          # Answer Set Prolog implementation of the ad hoc agent after refinment.
├── ASP/ahagent_pre.sp      # Answer Set Prolog implementation of the ad hoc agent.
├── ASP/huamn.sp            # Answer Set Prolog implementation of the human after refinment.
├── ASP/human_pre.sp        # Answer Set Prolog implementation of the human.
├── Execution Video         # Example execution video from the VirtualHome domain where a human and the ad hoc agents are collaborating to perform household tasks.
├── Explanations            # Code and results(with examples) for ad hoc agents providing explanations of its behaviour.
├── Models                  # Behaviour models learned by the ad hoc agent for other agents.
├── simulation              # Files from the VirtualHome domain with modification.
├── main.py                 # Main file
├── state.csv               # Data used for creating the behaviour model.
└── utils.py                # Utility file.
```
