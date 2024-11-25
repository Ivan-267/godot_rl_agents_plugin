extends Resource
class_name NeuralNetwork


class LinearLayer:
	var weights: Array[Array]
	var bias: Array[float]

	func create(input_size: int, output_size: int) -> LinearLayer:
		var layer := LinearLayer.new()
		layer.weights.resize(output_size)
		layer.bias.resize(output_size)
		layer.bias.fill(0)

		for output_idx in weights.size():
			layer.weights[output_idx].resize(input_size)
			layer.weights[output_idx].fill(0)

		return layer

	func calculate_output(input: Array[float]) -> Array[float]:
		assert(
			not weights.is_empty(),
			"The weights array is empty. Please populate it / initialize the parameters."
		)
		assert(
			not bias.is_empty(),
			"The bias array is empty. Please populate it / initialize the parameters."
		)

		var output: Array[float]
		output.resize(weights.size())
		output.fill(0)

		for output_idx in weights.size():
			for input_idx in weights[output_idx].size():
				output[output_idx] += input[input_idx] * weights[output_idx][input_idx]
			output[output_idx] += bias[output_idx]
		return output


## Applies tanh to each member of the array
class TanhLayer:
	func calculate_output(input: Array[float]) -> Array[float]:
		var output: Array[float]
		output.resize(input.size())
		output.fill(0)

		for i in input.size():
			output[i] = tanh(input[i])
		return output


## Allows to stack multiple layers
class Sequential:
	var layers: Array

	func add_layer(layer):
		layers.append(layer)

	func calculate_output(input: Array[float]) -> Array[float]:
		var output := input
		for layer in layers:
			output = layer.calculate_output(output)
		return output
