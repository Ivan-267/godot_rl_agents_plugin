extends Sync
class_name SyncGdscriptInference

## Overrides "onnx inference" mode to use gdscript-based inference.
##
## DISCLAIMER:
## Not tested much yet, inacurrate inference results or other errors might happen.
## Not recommended for production, no warranties of any kind provided.
##
## Current limitations:
## - Only a single policy for all AIControllers can be used for now.
## - Not multi-threaded or optimized, it can slow down the project. Use onnx inference if possible.

## JSON parameters exported from a supported training framework
@export var json_params: Resource

var policy: NeuralNetwork.Sequential
var single_agent_action_space: Dictionary

## Run a small, non-comprehensive suite of tests on _ready() to check observation type/size.
@export var run_tests: bool = true
## Print a message for each test that was successful.
@export var print_test_passed_messages: bool = true


func _ready() -> void:
	# Re-implementing super._ready() here so that the rest of _ready() won't be executed
	# before the await statements have finished waiting
	await get_parent().ready
	get_tree().set_pause(true)
	_initialize()
	await get_tree().create_timer(1.0).timeout
	get_tree().set_pause(false)

	if control_mode == ControlModes.ONNX_INFERENCE:
		assert(
			not agents_inference.is_empty(),
			"There are no AIControllers with 'inherit from sync' or 'onnx inference' control mode set."
		)
		var neural_network_loader := NeuralNetworkLoader.new()
		policy = neural_network_loader.load_sb3_single_hidden_layer_policy_from_json(json_params)
		single_agent_action_space = agents_inference[0].get_action_space()
		if run_tests:
			_run_tests()


func _run_tests():
	_test_obs_array_type()
	_test_obs_array_size()


# Check obs type. Correct type is Array[float]
func _test_obs_array_type():
	var test_obs = _get_obs_from_agents(agents_inference)[0]["obs"]
	if not test_obs is Array[float]:
		assert(
			false,
			(
				"get_obs() must return Array[float] type for gdscript inference.\n"
				+ "If you are returning an Array as observations, change it to Array[float], e.g. obs: Array[float]"
			)
		)
	else:
		if print_test_passed_messages:
			print_test_passed_message("Inference obs array type")


# Check that the size of the obs matches the size of the loaded JSON params
func _test_obs_array_size():
	var test_obs = _get_obs_from_agents(agents_inference)[0]["obs"]
	if not test_obs.size() == policy.layers[0].weights[0].size():
		assert(
			false,
			(
				"There's a mismatch between the obs size and what is expected from json_params set on sync node. \n"
				+ "Some possible causes: JSON params are somehow corrupt, or from another model / maybe the obs were changed since the model was trained."
			)
		)
	else:
		if print_test_passed_messages:
			print_test_passed_message("Inference obs array size")


func print_test_passed_message(test_type: String) -> void:
	print_rich(
		(
			"[color=green]%s test PASS. You can disable tests or these messages in sync node. [/color]"
			% [test_type]
		)
	)


func _initialize_inference_agents():
	_set_heuristic("model", agents_inference)


func _inference_process():
	if agents_inference.size() > 0:
		var obs: Array = _get_obs_from_agents(agents_inference)
		var actions = []

		for agent_id in range(0, agents_inference.size()):
			var action = policy.calculate_output(obs[agent_id]["obs"])
			var action_dict = _extract_action_dict(action, single_agent_action_space, 1)
			actions.append(action_dict)

		_set_agent_actions(actions, agents_inference)
		_reset_agents_if_done(agents_inference)
		get_tree().set_pause(false)
