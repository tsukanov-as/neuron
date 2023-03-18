package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/tsukanov-as/neuron"
)

var sample string

func init() {
	flag.StringVar(&sample, "sample", "", "-sample='aaaabbbbbbbbbcc'")
	flag.Parse()
}

func main() {

	const abc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

	const statesCount = 3

	// Распознаватель регулярного языка определенного выражением: ab*(c|d)
	// Фактически это автомат Глушкова, построенный на нейронах.

	// Нейроны OR

	// зависимости состояния 0

	c1_0 := neuron.New(2, statesCount+len(abc))

	v := make([]float64, statesCount+len(abc))
	v[0] = 0.0
	v[1] = 0.0
	v[2] = 0.0
	c1_0.Learn(0, v)

	v = make([]float64, statesCount+len(abc))
	v[statesCount+strings.IndexByte(abc, 'a')] = 1.0
	c1_0.Learn(1, v)

	// зависимости состояния 1

	c1_1 := neuron.New(2, statesCount+len(abc))

	v = make([]float64, statesCount+len(abc))
	v[0] = 1.0
	v[1] = 1.0 // зависит от самого себя
	v[2] = 0.0
	c1_1.Learn(0, v)

	v = make([]float64, statesCount+len(abc))
	v[statesCount+strings.IndexByte(abc, 'b')] = 1.0
	c1_1.Learn(1, v)

	// зависимости состояния 2

	c1_2 := neuron.New(2, statesCount+len(abc))

	v = make([]float64, statesCount+len(abc))
	v[0] = 1.0
	v[1] = 1.0
	v[2] = 0.0
	c1_2.Learn(0, v)

	v = make([]float64, statesCount+len(abc))
	v[statesCount+strings.IndexByte(abc, 'c')] = 1.0
	v[statesCount+strings.IndexByte(abc, 'd')] = 1.0
	c1_2.Learn(1, v)

	// Нейроны AND

	c2 := neuron.New(statesCount, statesCount*2)

	state := 0
	v = make([]float64, statesCount*2)
	v[state*2+0] = 0.0 // не зависит от других состояний
	v[state*2+1] = 1.0
	c2.Learn(state, v)

	state = 1
	v = make([]float64, statesCount*2)
	v[state*2+0] = 1.0
	v[state*2+1] = 1.0
	c2.Learn(state, v)

	state = 2
	v = make([]float64, statesCount*2)
	v[state*2+0] = 1.0
	v[state*2+1] = 1.0
	c2.Learn(state, v)

	// Сопоставление

	states := make([]float64, statesCount)

	for i := 0; i < len(sample); i++ {
		var chars [len(abc)]float64
		char := sample[i]
		chars[strings.IndexByte(abc, char)] = 1.0
		features := append(states, chars[:]...)
		s0, err := c1_0.Predict(features)
		if err != nil {
			log.Fatal(err)
		}
		s1, err := c1_1.Predict(features)
		if err != nil {
			log.Fatal(err)
		}
		s2, err := c1_2.Predict(features)
		if err != nil {
			log.Fatal(err)
		}
		var interim []float64
		interim = append(interim, s0...)
		interim = append(interim, s1...)
		interim = append(interim, s2...)
		states, err = c2.Detect(interim)
		if err != nil {
			log.Fatal(err)
		}
		if states[2] > 0 {
			fmt.Println("matched!")
			return
		}
	}
	fmt.Println("not matched")
}
