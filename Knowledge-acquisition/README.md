# Knowledge Acquisition in Ad Hoc Teamwork

Ad hoc teamwork involves agents collaborating with teammates without prior coordination. Achieving such teamwork is challenging as agents may have differing and incomplete knowledge about the environemnt.

This repository contains the code for an architecture that enables an ad hoc agent to incrementally acquire and refine domain knowledge to improve collaboration and task execution. The approach combines:

- Answer Set Prolog (ASP) for representing domain knowledge.
- A Large Language Model (LLM) to interpret human-provided natural language descriptions of agent behaviors, enabling the extraction of missing knowledge in terms of sorts, actions and axioms.
- Decision tree induction for learning previously unknown causal laws and executability conditions.
- Ecological rationality inspired beahviour prediction models of other agents.

We use VirtualHome, a 3D simulation environment to evalute the architecture.

## Folder Structure

```bash
.
├── ASP/ahagent1.sp             # Answer Set Prolog implementation of the ad hoc agent with comprehensive knowledge after refinment.
├── ASP/ahagent_pre1.sp         # Answer Set Prolog implementation of the ad hoc agent with comprehensive knowledge.
├── ASP/ahagent2.sp             # Answer Set Prolog implementation of the ad hoc agent with incomplete knowledge after refinment.
├── ASP/ahagent_pre2.sp         # Answer Set Prolog implementation of the ad hoc agent with incomplete knowledge.
├── ASP/human.sp                # Answer Set Prolog implementation of the human after refinment.
├── ASP/human_pre.sp            # Answer Set Prolog implementation of the human.
├── ASP/asp_sorts.txt           # Sorts for the comprehensive knowledge agent.
├── ASP/asp_sorts_partial.txt   # Sorts for the incomplete knowledge agent.
├── ASP/sparc.jar               # ASP SPARC jar file
├── Decision Tree Induction     # Code for acquiring knowledge through decision tree induction.
├── Models                      # Behaviour models learned by the ad hoc agent for other agents.
├── simulation                  # Modified files from the VirtualHome domain.
├── main.py                     # Main file
├── utils.py                    # Utility file for comprehensive knowledge agent.
└── utils_partial.py            # Utility file for incomplete comprehensive knowledge agent.
```
## Installation

Follow the instructions from [VirtualHome](http://virtual-home.org) to download and install VirtualHome.

Replace the corresponding files in the installed directory with those from the virtualhome/simulation/unity_simulator folder.

Use the main.py file to run the code for proposed architecture.
