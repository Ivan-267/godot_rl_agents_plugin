## Experimental/testing gdscript inference branch:

This is an experimental branch that enables using gdscript inference instead of C# inference.
It is based on the [Gdscript Neural Network Experiment](https://github.com/Ivan-267/GdscriptNeuralNetExperiment) repository (MIT licensed) with 
additional code added to load parameters from SB3 trained models, and run inference with GDRL.

Note that this is a very basic gdscript implementation, it might work slow, have bugs, or etc. No warranties are provided. If possible, you should use C# inference instead.

## Usage:
[To be added]

# Godot RL Agents

This repository contains the Godot 4 asset / plugin for the Godot RL Agents library, you can find out more about the library on its Github page [here](https://github.com/edbeeching/godot_rl_agents).

The Godot RL Agents is a fully Open Source package that allows video game creators, AI researchers and hobbyists the opportunity to learn complex behaviors for their Non Player Characters or agents. 
This libary provided this following functionaly:
* An interface between games created in the [Godot Engine](https://godotengine.org/) and Machine Learning algorithms running in Python
* Wrappers for three well known rl frameworks: StableBaselines3, Sample Factory and [Ray RLLib](https://docs.ray.io/en/latest/rllib/index.html)
* Support for memory-based agents, with LSTM or attention based interfaces
* Support for 2D and 3D games
* A suite of AI sensors to augment your agent's capacity to observe the game world
* Godot and Godot RL Agents are completely free and open source under the very permissive MIT license. No strings attached, no royalties, nothing. 

You can find out more about Godot RL agents in our AAAI-2022 Workshop [paper](https://arxiv.org/abs/2112.03636).

