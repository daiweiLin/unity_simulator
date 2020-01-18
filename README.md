# LAS_Unity
Living Architecture System simulated environment with Unity's [Machine Learning Toolkits](https://github.com/Unity-Technologies/ml-agents).
## Installation
Refer to Unity's instructions: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
If you want to install old versions of toolkits instead of the latest version, for example v0.6 used in this repository, do following instead
```
pip install mlagents=0.6.0
```
## To run the simulating environment
### Method 1: Inside Unity editor
1. When initialize the `env` object, set `file_name=None`
   ```python
      env = UnityEnvironment(file_name=None, seed=1)
   ```
2. Then click START button in editor to start the simulation

### Method 2: Using Unity executables 
   1. Export the Unity project as executables. [Instruction](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Executable.md). For Windows users, files are saved in a folder.
   2. Set file_name equal to the path to the folder.
   ```python
      env_name = 'LAS-Scenes/Unity/LAS_Simulator'
      env = UnityEnvironment(file_name=env_name, seed=1)
   ```
This will run simulation in the Unity executables. The simulation usually is faster using executables than inside the Unity editor.

## Organization
### Composition
   1. Interaction between LAS, visitors and Environment: `Test_simulator.py`
   2. Prescripted Behaviour: `Prescripted_behaviour_timing.py`. This file uses time in Unity simulator instead of world time, so that pre-scripted behaviour is not affected by simulation speed.
   3. Visitor:
      * `Visitor_behaviour.py` --- Attracted by Intensity of LED at each time step
      * `Visitor_behaviour_sequence.py` --- Attracted by a sequence of actions produced by LAS
   4. LAS agents: All in `LASAgent` folder
   

### Interaction paradigm and Simulator

|Overall Diagram           | Adaptive Behaviour Block |
:-------------------------:|:-------------------------:
|![](https://github.com/daiweiLin/unity_simulator/blob/master/InitialDesignIdeas/DesignFigures/Interaction%20Diagram.png)  |  ![](https://github.com/daiweiLin/unity_simulator/blob/master/InitialDesignIdeas/DesignFigures/Adaptive%20Behaviour.png)|


## Dependency
   1. OpenAI baseline, spinningup
   2. Unity's Machine Learning Toolkit
   3. Tensorflow
