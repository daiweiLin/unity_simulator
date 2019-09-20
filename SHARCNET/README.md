# Sharcnet Installation Instructions
The following is the instructions on how to install Unity ML Toolkits on Sharcnet.

## 1. Create virtual environment and install Tensorflow 1.7
[Reference of sharknet](https://docs.computecanada.ca/wiki/TensorFlow)

As ML toolkits use python 3.6, load python 3.6 first
```
[name@server ~]$ module load python/3.6
```
Create a new Python virtual environment:
```
[name@server ~]$ virtualenv unity_ml
```
Activate your newly created Python virtual environment:
```
[name@server ~]$ source unity_ml/bin/activate
```
Install Tensorflow 1.7, this is a version customized by computecanada:
```
[name@server ~]$ pip install tensorflow_cpu==1.7.0+computecanada
```
After installation is complete, if you run `python -c "import tensorflow as tf; print(tf.__version__)"`, you will see `1.7.0`

## 2. Download Unity's Toolkit (version 0.6)
[Installation Reference from Unity's ML Toolkits](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

Clone repository and checkout the old version:
```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
git checkout 3b77fe9f9194ca504f57c1151c64c884d7d3f025
```
Note: The version `3b77fe9f9194ca504f57c1151c64c884d7d3f025` is published on Feb 5th, 2019. See https://github.com/Unity-Technologies/ml-agents/commits for commit histories.
## 3. Manual Install
As ml-agent only works for tensorflow 1.7, and the version number installed in this case is `1.7.0+computecanada`, the setup file cannot find tensorflow properly. So we comment out the version check from `setup.py`

Go to the ml-agents/ml-agents and open setup.py:
```
(unity_ml)_[name@server ~]$ cd ml-agents/ml-agents
(unity_ml)_[name@server ~]$ vim setup.py
```
Comment out `'tensorflow>=1.7,<1.8',` under `install_requires`.
```python
install_requires=[
    'mlagents_envs==0.8.1',
    #'tensorflow>=1.7,<1.8',
    'Pillow>=4.2.1',
    'matplotlib',
    'numpy>=1.13.3,<=1.14.5',
    'jupyter',
    'pytest>=3.2.2,<4.0.0',
    'docopt',
    'pyyaml',
    'protobuf>=3.6,<3.7',
    'grpcio>=1.11.0,<1.12.0',
    'pypiwin32==223;platform_system=="Windows"'],
```
Save and close the file. Then go to the parent folder `ml-agents`. For each folder `ml-agents`,`gym-unity`, install manually.
```
cd ml-agents
pip install -e ./
cd ..
cd gym-unity
pip install -e ./
```
Notice that when installing ml-agents, the numpy version is required to be lower than 1.14.5. This is not supported by tensorflow1.7.0+computecanada. Therefore, after the installation above is finished, force update the numpy once again. This won't affect ml-agents (at least at the moment).
```
pip install numpy --upgrade
```

## 4. Install Baselines
See [here](https://github.com/UWaterloo-ASL/ML_lite/blob/master/README.md) for installation detail
## 5. Install SpinningUp

```
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```
See [SpinningUp](https://spinningup.openai.com/en/latest/user/installation.html) for more details

# Download logs from SHARCNET
Open a local session on MobaXterm, and open the home directory:
```linux
open /home/mobaxterm
```
Put download_log*.sh files in the directory. Then run the scripts like following
```
bash dowload_log.sh
```
This will download the files into MobaXterm's home directory under **downloads** folder, for example C:\Users\YourName\Documents\MobaXterm\home\downloads
