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

func TestAssociation(t *testing.T) {
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

	// fmt.Println(c.ClassProbs(1))
	// fmt.Println(c.FeatureProbs(2))
}

func complemented(x []float64) []float64 {
	lim := len(x)
	for i := 0; i < lim; i++ {
		x = append(x, 1-x[i])
	}
	return x
}

func TestDetection(t *testing.T) {
	// Детектор хорошо работает только с заведомо слабо пересекающимся по признакам классам.
	// Для каждой комбинации слабо пересекающихся классов (либо просто одного класса) нужен отдельный детектор.
	// Кроме того, необходимо добавить инвертированные признаки, чтобы детектор "видел" отсутствие признака.
	x := []rec{ // класс, признаки
		{0, complemented([]float64{0, 1, 1, 0})},
		{1, complemented([]float64{1, 0, 0, 1})},
	}

	c := New(2, 4*2)

	for _, r := range x {
		err := c.Learn(r.cl, r.fv)
		if err != nil {
			t.Fatal(err)
		}
	}

	p, err := c.Detect(complemented([]float64{0, 0.1, 0.8, 0}))
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(argmax(p), p)
}

func TestDetection2(t *testing.T) {
	// Адаптивный детектор
	x := []rec{ // класс, признаки
		{0, complemented([]float64{0, 1, 1, 0})},
		{1, complemented([]float64{1, 0, 0, 1})},
	}

	c := New(2, 4*2)

	for i := 0; i < 10000; i++ {
		for _, r := range x {
			err := c.Learn(r.cl, r.fv)
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	p, err := c.Detect2(complemented([]float64{0, 0.1, 0.8, 0}))
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(argmax(p), p)
}
