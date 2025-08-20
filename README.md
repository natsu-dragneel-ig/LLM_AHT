# Reasoning with Commonsense Knowledge and Heuristically-learned Models for Ad hoc Human - Agent Collaboration

AI agents deployed to cooperate with others are often required to do so without prior coordination. State of the art approaches for such ad hoc teamwork pose this task as a learning problem, using a large labeled dataset to model the action choices of other agents (or agent types) and determine the actions of the ad hoc agent. These methods lack transparency and make it difficult to rapidly revise existing knowledge in response to changes. We present an architecture for ad hoc teamwork that leverages the complementary strengths of knowledge-based and data-driven methods for reasoning and learning. For any given goal, the ad hoc agent determines its actions through non-monotonic logical reasoning with: 
- prior domain specific commonsense knowledge;
- models learned and revised rapidly to predict the behavior of other agents; and
- anticipated abstract future goals based on generic knowledge of similar situations in an existing foundation model.

The agent also processes natural language descriptions and observations of other agents’ behavior, incrementally acquiring and revising knowledge in the form of objects, actions, and axioms that govern domain dynamics.
We experimentally evaluate the capabilities of our architecture in VirtualHome, a realistic simulation environment.

## Video
Example videos of human and ad hoc agents collaborating together to perform household tasks:

Video showing three agents-(two ad hoc agents and one human) collaborating to complete a series of household tasks.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/RRnCDx6D4zc/0.jpg)](https://www.youtube.com/watch?v=RRnCDx6D4zc)


Video showing four agents-(three ad hoc agents and one human) collaborating to complete a series of household tasks.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/l-4D-LtNX8k/0.jpg)](https://www.youtube.com/watch?v=l-4D-LtNX8k)

The action selection of ad hoc agent enhances team collaboration and prevents conflicts, enabling the agents to achieve their goal more efficiently and within less time.
Moreover, the architecture scales efficiently to accommodate a larger number of agents, leading to improved performance.

## Providing Explanations of Decisions
The use of knowledge-based reasoning and simple predictive models provide the ability for the ad hoc agent to generate relational descriptions in response to four types of questions: Causal, Contrastive, Justify beliefs and Counterfactual Questions. The performance of the architecture when creating these explanations can be found here together with execution examples: [Explanations](https://github.com/hharithaki/Task-Anticipation/tree/main/Explanations)

## Results on Scalability
Following table shows the average number of steps and time taken by different teams, Team1 (human, ad hoc agent), Team2 (human, two ad hoc agents) and Team3 (human, three ad hoc agents) to complete 100 task routines when collaborating together. The architecture scale to different number of agents efficiently improving performance.

|            Agent Team           |   Steps   | Time(s) |
| :------------------------------ | :-------: | -----:  |
| Team1 (human + 1 ad hoc agent)  |    26.8   |  361.0  |
| Team2 (human + 2 ad hoc agents) |    22.8   |  329.4  |
| Team3 (human + 3 ad hoc agents) |    19.5   |  307.9  |

## Acquiring new Knowledge
The ad hoc agent incrementally revise prior knowledge based on LLM-based processing of natural language descriptions of actions and outcomes, and decision tree induction applied to observations.
Source code can be found here: [Knowledge-acquisition](https://github.com/hharithaki/Task-Anticipation/tree/main/Knowledge-acquisition)

## Folder Structure

```bash
.
├── ASP/ahagent.sp          # Answer Set Prolog implementation of the ad hoc agent after refinment.
├── ASP/ahagent_pre.sp      # Answer Set Prolog implementation of the ad hoc agent.
├── ASP/human.sp            # Answer Set Prolog implementation of the human after refinment.
├── ASP/human_pre.sp        # Answer Set Prolog implementation of the human.
├── Explanations            # Source code and results(with examples) for ad hoc agents providing explanations of its behaviour when using KAT.
├── Knowledge-acquisition   # Source code for acquiring new knowledge by the ad hoc agent when using KAT.
├── Execution_traces.pdf    # Execution traces demonstrating ad hoc agent's performance when using the task anticipation component.
├── Models                  # Behaviour models learned by the ad hoc agent for other agents.
├── simulation              # Files from the VirtualHome domain with modification.
├── main.py                 # Main file.
└── utils.py                # Utility file.

```
