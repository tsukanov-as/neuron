package neuron

// wire with negative probability
func not(a float64) float64 {
	return 1 - a
}

// 1 neuron in modification `or`
func or(a, b float64) float64 {
	return a + b - a*b
}

// 1 neuron in modification `and`
func and(a, b float64) float64 {
	return a * b
}

// 1 neuron in modification `xor` and negative wire (or negative modification of neuron)
func xnor(a, b float64) float64 {
	return not(xor(a, b))
}

// 1 neuron in modification `xor`
func xor(a, b float64) float64 {
	return not(a)*b + a*not(b)
}

func imply(a, b float64) float64 {
	return or(not(a), b)
}
