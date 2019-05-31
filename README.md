# LAS_Unity
Living Architecture System simulated environment with Unity's [Machine Learning Toolkits](https://github.com/Unity-Technologies/ml-agents).

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
### Interaction scripts
   1. Interaction between LAS, visitors and Environment: `Test_simulator.py`
   2. Prescripted Behaviour: `Prescripted_behaviour.py`

### Interaction paradigm and Simulator
<img src="https://github.com/daiweiLin/unity_simulator/blob/master/InitialDesignIdeas/DesignFigures/Interaction%20Diagram.png" /> 

## Dependency
   1. OpenAI baseline
   2. Unity's Machine Learning Toolkit
   3. Tensorflow
   4. tflearn
