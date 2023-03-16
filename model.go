package neuron

import (
	"fmt"
	"math/rand"
)

// DRAFT

type Type int

const (
	OR  Type = 0
	XOR Type = 1
	AND Type = 2
)

const CassetteSize = 10

type Neuron struct {
	Label    string
	Type     Type
	Stress   float64
	Limit    float64
	Weight   float64
	Timer    int
	Cassette int
	Incoming []*Wire
	Outgoing []*Wire
}

func NewNeuron(t Type, label string, w float64) *Neuron {
	return &Neuron{
		Label:  label,
		Weight: w,
		Type:   t,
		Limit:  CassetteSize,
	}
}

func (n *Neuron) DependsOn(src *Neuron, w float64) {
	wire := &Wire{
		Weight: w,
		Src:    src,
		Dst:    n,
	}
	n.Incoming = append(n.Incoming, wire)
	src.Outgoing = append(src.Outgoing, wire)
}

func (n *Neuron) Kick() {
	n.Cassette = CassetteSize
}

func (n *Neuron) Tick() {
	sum := 0
	for _, w := range n.Incoming {
		if w.Impulse {
			w.Impulse = false
			sum++
		}
	}
	switch n.Type {
	case OR:
		if sum > 0 {
			n.Stress++
		}
	case XOR:
		if sum == 1 {
			n.Stress++
		}
	case AND:
		if sum > 0 && sum == len(n.Incoming) {
			n.Stress++
		}
	}
	impulse := false
	n.Timer++
	if n.Timer > CassetteSize {
		n.Timer = 0
		prob := n.Stress / n.Limit
		GodDice := rand.Float64()
		if GodDice < prob {
			impulse = true
			fmt.Printf("%s fired! (stress: %.0f)\n", n.Label, n.Stress)
		}
		n.Stress = 0
	}
	if impulse {
		n.Cassette = CassetteSize // load guns
	}
	for _, w := range n.Outgoing {
		w.Tick()
	}
	if n.Cassette > 0 {
		n.Cassette-- // completed one volley from all guns with corresponding probabilities
	}
}

type Wire struct {
	Impulse bool
	Weight  float64
	Src     *Neuron
	Dst     *Neuron
}

func (w *Wire) Tick() {
	if w.Src.Cassette > 0 {
		prob := w.Weight / w.Src.Weight
		GodDice := rand.Float64()
		if GodDice < prob {
			w.Impulse = true
			fmt.Printf("wire: %s -> %s = 1\n", w.Src.Label, w.Dst.Label)
		} else {
			fmt.Printf("wire: %s -> %s = 0\n", w.Src.Label, w.Dst.Label)
		}
	}
}

type TwoLayerNetwork struct {
	Input  []*Neuron
	Output []*Neuron
}

func (nn TwoLayerNetwork) Tick() {
	for _, n := range nn.Input {
		n.Tick()
	}
	for _, n := range nn.Output {
		n.Tick()
	}
}
