## Experimental/testing gdscript inference branch:

This is an experimental branch that enables using gdscript inference instead of C# inference.
It is based on the [Gdscript Neural Network Experiment](https://github.com/Ivan-267/GdscriptNeuralNetExperiment) repository (MIT licensed) with 
additional code added to load parameters from SB3 trained models, and run inference with GDRL.

Note that this is a very basic gdscript implementation, it might work slow, have bugs, or etc. No warranties are provided. If possible, you should use C# inference instead.

## Usage:
### 1 - Saving JSON params during training:
You can modify the SB3 training example script, or other scripts, by adding the following (e.g. at the end of the SB3 example):
```Python
# -- Save the policy parameters to JSON (tested with TRPO, should work with PPO and similar) --
state_dict_json = {key: value.tolist() for key, value in model.policy.state_dict().items()}

with open('params_sb3.json', 'w') as f:
    json.dump(state_dict_json, f)
# --
```
You can also adjust the file path above.

### 2 - Copying the files and Replacing `sync` node:
- It's recommended to make a backup of your project with the standard C# inference, just in case.

- Copy the `neural_network` folder into your game project. You can also copy the addon folder if you haven't already loaded the GDRL plugin or if you want to use the version here for compatibility. (it's only tested with the version from this repository, and should work with [this version of GDRL](https://github.com/edbeeching/godot_rl_agents/tree/e1d89c99cb78a0224de23a408aec1fe99679a7e9), may or may not work.

- Drag and drop the `neural_network > sync_gdscript_inference` node into your `Sync` node to replace the script.

![image](https://github.com/user-attachments/assets/aac1e649-6ad5-4d09-8dd3-a7236d34fe72)

- Copy your trained .JSON parameters to your Godot project, then drag and drop them into the `Json Params` property in the inspector

![image](https://github.com/user-attachments/assets/cf726b1f-ed92-4913-8650-e90c23752d21)

- Check that the `Control Mode` is set to `Onnx Inference`. `SyncGdscriptInference` overrides this mode such that the JSON params are used and gdscript based inference is used, `Onnx Model Path` is ignored.

Limitations: Beside the perfomance limitations mentioned, currently this only supports a single policy for AI Controllers, you can't use separate policies for each AIController.

### 3 - Adjusting inference to load the params based on the current model:
- Open the script for the sync node (should be the `SyncGdscriptInference` script):
- Find this line `policy = neural_network_loader.load_sb3_single_hidden_layer_policy_from_json(json_params)` and adjust it based on your policy. There are methods for loading a few different SB3 policies, here is how to export for each one:

`neural_network_loader.load_sb3_single_hidden_layer_policy_from_json(json_params)`: Loads a policy trained with a single hidden layer in SB3. You can achieve this by adjusting your model params in the SB3 training script to e.g.
```Python
    model: PPO = PPO(
        "MultiInputPolicy",
        env,
        verbose=2,
        n_steps=32,
        tensorboard_log=args.experiment_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[8], vf=[64, 64]),
        )
    )
```
`neural_network_loader.load_sb3_mlp_policy_from_json(json_params)`: Loads the standard MLP in SB3, you don't need to set any policy kwargs.
`neural_network_loader.load_sb3_linear_policy_from_json(json_params)`: Loads a policy trained without hidden layers, e.g:
```Python
    model: PPO = PPO(
        "MultiInputPolicy",
        env,
        verbose=2,
        n_steps=32,
        tensorboard_log=args.experiment_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[], vf=[64, 64]),
        )
    )
```
To import policies trained with other frameworks (e.g. CleanRL), or use a different number of layers, you can add your own loading function in `neural_network_loader.gd`, make sure to check the names used for weight and biases in the exported JSON.
Compatability with other frameworks is not guaranteed.

Note that it's best to use policies as simple/small as possible as the inference optimization is not optimized. Where possible, try linear. If not, try a single hidden layer network with a small number of neurons, etc.
Again, this is more of a simple implementation for testing gdscript based inference rather than an optimized implementation, slowdowns or bugs can happen, so it's not recommended for production use.

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

