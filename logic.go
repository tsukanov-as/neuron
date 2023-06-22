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

func xnor(a, b float64) float64 {
	return not(xor(a, b))
}

func xor(a, b float64) float64 {
	return or(not(a)*b, a*not(b))
}
