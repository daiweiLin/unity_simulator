# LASAgent classes
This folder contains implemented Learning Algorithms for Living Architecture System. In theory, **Living Architecture Sytem** and **Visitor** could share the same set of Learning Algorithms to realize control of intelligent agent. However, in practice, using learning algorithm to control visitor is very hard and complex. Therefore, in our implementation, we separatively maintain control algorithms for [Living Architecture System](https://github.com/UWaterloo-ASL/LAS_Gym/tree/master/LASAgent) and [Visitor](https://github.com/UWaterloo-ASL/LAS_Gym/tree/master/VisitorAgent).

## Intermediate Internal Environment Classes
To ensure reusability, we use an intermediate class for realistic interaction in which reward signal is not provided by the environment, at the same time seamlessly working with virtual environment with interfaces as in [OpenAI Gym](https://gym.openai.com/docs/). 
**Internal Environment for Single Agent** is mainly used to receive observation from, calculate reward and deliver action chosen by agent to real or virtual external environment.
   * InternalEnvOfAgent.py

## Learning Agent Classes

We adapte implemenations of **DDPG**, **PPO** and **TD3** from OpenAI's [Baseline](https://github.com/openai/baselines) and [SpinUp](https://spinningup.openai.com/en/latest/#) libraries.
Baseline library version:
  * DDPG - `LASBaselineAgent.py`

SpinUp library version:
  * DDPG - `LASSpinUpDDPGAgent.py`
  * PPO - `LASSpinUpPPOAgent.py`
  * TD3 - `LASSpinUpTD3Agent.py`
 
We also have an agent that takes random actions at each time step.
   * Implememted in `RandomLASAgent.py`
