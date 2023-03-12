package neuron

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// DRAFT

func TestModel(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	feature1 := NewNeuron(AND, "feature1", 50)  // 10 for class1 and 40 for class2
	feature2 := NewNeuron(AND, "feature2", 100) // 50 for class2 and 50 for other classes

	class1 := NewNeuron(OR, "class1", 0 /* doesn't matter */) // Pr = 10/50 = 0.2
	class1.DependsOn(feature1, 10)                            // Pr = 10/50

	class2 := NewNeuron(OR, "class2", 0 /* doesn't matter */) // Pr = 40/50 + 50/100 - 40/50 * 50/100 = 0.9
	class2.DependsOn(feature1, 40)                            // Pr = 40/50
	class2.DependsOn(feature2, 50)                            // Pr = 50/100

	nn := TwoLayerNetwork{
		Input: []*Neuron{
			feature1,
			feature2,
		},
		Output: []*Neuron{
			class1,
			class2,
		},
	}

	feature1.Kick()
	feature2.Kick()

	for i := 0; i < CassetteSize+1; i++ {
		fmt.Println("tick: ", i)
		nn.Tick()
	}
}
