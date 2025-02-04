# Generic-to-Specific Reasoning and Ad Hoc Teamwork in Embodied AI Agents

Embodied AI agents deployed in assistive roles often have to collaborate with other agents (humans, AI systems) without prior coordination. Methods considered state of the art for such ad hoc teamwork often pursue a data-driven approach that needs a large labeled dataset of prior observations, lacks transparency, and makes it difficult to rapidly revise existing knowledge in response to changes. This repository contains the supplementary files for a hybrid architecture for ad hoc teamwork leveraging the complementary strengths of knowledge-based and data-driven methods for reasoning and learning for ad hoc teamwork. 

Our architecture enables an ad hoc agent to determine its actions through non-monotonic logical reasoning at different abstractions with: 
- a. prior commonsense domain-specific knowledge
- b. models learned and revised rapidly to predict the behavior of other agents
- c. anticipated abstract future goals based on generic knowledge of similar domains in an existing Foundation Model.

We use VirtualHome, a 3D simulation environment to evalute the architecture.

Example videos of human and ad hoc agents collaborating together to perform household tasks: [Execution Video](https://github.com/natsu-dragneel-ig/LLM_AHT/tree/main/Execution%20Video)

The use of knowledge-based reasoning and simple predictive models provide the ability for the ad hoc agent to generate relational descriptions in response to four types of questions: Causal, Contrastive, Justify beliefs and Counterfactual Questions. The performance of the architecture when creating these expelnation can be found here together with execution examples: [Explanations](https://github.com/natsu-dragneel-ig/LLM_AHT/tree/main/Explanations)

## Folder structure

```bash
.
├── ASP/ahagent.sp          # Answer Set Prolog implementation of the ad hoc agent after refinment.
├── ASP/ahagent_pre.sp      # Answer Set Prolog implementation of the ad hoc agent.
├── ASP/huamn.sp            # Answer Set Prolog implementation of the human after refinment.
├── ASP/human_pre.sp        # Answer Set Prolog implementation of the human.
├── Execution Video         # Example execution video from the VirtualHome domain where a human and ad hoc agents are collaborating to perform household tasks.
├── Explanations            # Code and results(with examples) for ad hoc agents providing explanations of its behaviour.
├── Models                  # Behaviour models learned by the ad hoc agent for other agents.
├── simulation              # Files from the VirtualHome domain with modification.
├── main.py                 # Main file
├── state.csv               # Data used for creating the behaviour model.
└── utils.py                # Utility file.
```
