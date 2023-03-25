package neuron

import (
	"fmt"
	"strings"
	"testing"
)

func TestWordProximity(t *testing.T) {
	const abc = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
	maxlen := 0

	encode := func(w string) []float64 {
		v := make([]float64, len([]rune(w))*len(abc))
		i := 0
		for _, r := range w {
			v[len(abc)*i+strings.IndexRune(abc, r)] = 1.0
			i++
		}
		if len(v) > maxlen {
			maxlen = len(v)
		}
		return v
	}

	x := []rec{
		{0, encode("милка")},
		{1, encode("мыла")},
		{2, encode("мыло")},
		{3, encode("рыло")},
		{4, encode("милк")},
	}

	c := New(len(x), len(abc)*maxlen)

	for _, r := range x {
		c.Learn(r.cl, r.fv)
	}

	p, err := c.Predict(encode("мило"))
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(p)
}
