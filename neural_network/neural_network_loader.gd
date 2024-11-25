extends Resource
class_name NeuralNetworkLoader

## This class loads parameters for a NeuralNetwork from JSON

func create_tanh_layer() -> NeuralNetwork.TanhLayer:
	return NeuralNetwork.TanhLayer.new()


func load_linear_layers_from_json(
	file_resource: Resource, weight_names: Array[String], bias_names: Array[String]
) -> Array[NeuralNetwork.LinearLayer]:
	assert(file_resource is JSON, "file_resource should be a JSON object with the params")
	var params = file_resource.data

	var layers: Array[NeuralNetwork.LinearLayer]
	for index in weight_names.size():
		var layer := NeuralNetwork.LinearLayer.new()
		var json_weights = params[weight_names[index]]
		var json_bias = params[bias_names[index]]
		layer.weights.assign(json_weights.duplicate(true))
		layer.bias.assign(json_bias.duplicate())
		layers.append(layer)
	return layers


func load_sb3_mlp_policy_from_json(params_json: Resource):
	var policy := NeuralNetwork.Sequential.new()

	var linear_layers := load_linear_layers_from_json(
		params_json,
		[
			"mlp_extractor.policy_net.0.weight",
			"mlp_extractor.policy_net.2.weight",
			"action_net.weight"
		],
		["mlp_extractor.policy_net.0.bias", "mlp_extractor.policy_net.2.bias", "action_net.bias"],
	)

	policy.add_layer(linear_layers[0])
	policy.add_layer(create_tanh_layer())
	policy.add_layer(linear_layers[1])
	policy.add_layer(create_tanh_layer())
	policy.add_layer(linear_layers[2])
	return policy


func load_sb3_single_hidden_layer_policy_from_json(params_json: Resource):
	var policy := NeuralNetwork.Sequential.new()

	var linear_layers := load_linear_layers_from_json(
		params_json,
		["mlp_extractor.policy_net.0.weight", "action_net.weight"],
		["mlp_extractor.policy_net.0.bias", "action_net.bias"],
	)

	policy.add_layer(linear_layers[0])
	policy.add_layer(create_tanh_layer())
	policy.add_layer(linear_layers[1])
	return policy


func load_sb3_linear_policy_from_json(params_json: Resource):
	var policy := NeuralNetwork.Sequential.new()

	var linear_layers := load_linear_layers_from_json(
		params_json,
		["action_net.weight"],
		["action_net.bias"],
	)

	policy.add_layer(linear_layers[0])
	return policy
