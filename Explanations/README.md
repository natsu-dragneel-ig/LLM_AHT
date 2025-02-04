Our architecture enables the ad hoc agent to generate relational descriptions as explanations of its decisions and beliefs and those of the other agents.

For our experiments we designed different configurations in the VirtualHome each with a different arrangement and status of objects in the initial condition, 
**e.g.**, *bread on the kitchen table instead of the counter, microwave open instead of closed*.
We then measured the ability of the agent to come up with answers to 32 different questions (divided between the four types of questions- causal, contrastive, justify beliefs and counterfactuals) by recording the precision and recall values of retrieving the literals to answer these questions.

| Question type        | Precision | Recall |
| :------------------- | :-------: | -----: |
| Action justification |    1.00   |  1.00  |
| Contrastive          |    0.89   |  0.99  |
| Belief justification |    0.88   |  0.94  |
| Counterfactual       |    1.00   |  0.78  |

High values of precision and recall indicate the ability to automatically extract the correct literals to provide relational descriptions as explanations in response to different types of questions.
