package neuron

import (
	"fmt"
	"testing"
)

type rec struct {
	cl int
	fv []float64
}

func argmax(x []float64) int {
	j := 0
	max := 0.0
	for i := 0; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
			j = i
		}
	}
	return j
}

func TestStudent(t *testing.T) {
	x := []rec{ // класс, [очки, галстук, билет]
		// студент
		{0, []float64{1.0, 1.0, 1.0}},
		{0, []float64{1.0, 1.0, 0.0}},
		{0, []float64{1.0, 1.0, 0.0}},
		{0, []float64{1.0, 1.0, 0.0}},
		// не студент
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
	}

	c := New(2, 3)

	for _, r := range x {
		err := c.Learn(r.cl, r.fv)
		if err != nil {
			t.Fatal(err)
		}
	}

	p, err := c.Predict([]float64{1.0, 1.0, 0.0})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(argmax(p), p)
}
