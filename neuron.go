package neuron

import (
	"errors"
	"math"
)

type Classifier struct {
	cc  int       // classes count
	fc  int       // features count
	fs  []float64 // feature statistics by class
	ft  []float64 // features total
	ftt []float64 // features total
	ct  []float64 // classes total
	tc  float64
}

func New(classes, features int) *Classifier {
	c := &Classifier{
		cc:  classes,
		fc:  features,
		fs:  make([]float64, classes*features),
		ft:  make([]float64, features),
		ftt: make([]float64, classes),
		ct:  make([]float64, classes),
	}
	return c
}

func (c *Classifier) Init(cv float64, fv float64) error {
	if fv > cv {
		return errors.New("feature value must be less than or equal to class value")
	}
	for i := range c.ct {
		c.ct[i] = cv
	}
	for i := range c.fs {
		c.fs[i] = fv
	}
	for i := range c.ft {
		c.ft[i] = fv
	}
	for i := range c.ftt {
		c.ftt[i] = fv
	}
	return nil
}

func (c *Classifier) Learn(class int, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
	c.tc += 1
	c.ct[class] += 1
	base := class * c.fc
	fc := len(fv)
	if fc > c.fc {
		fc = c.fc
	}
	for fi := 0; fi < fc; fi++ {
		if fv[fi] < 0 || fv[fi] > 1 {
			return errors.New("feature value must be in range 0..1")
		}
		c.fs[base+fi] += fv[fi]
		c.ft[fi] += fv[fi]
		c.ftt[class] += fv[fi]
	}
	return nil
}

// experimental
func (c *Classifier) Learn2(class int, cv float64, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
	c.tc += cv
	c.ct[class] += cv
	base := class * c.fc
	fc := len(fv)
	if fc > c.fc {
		fc = c.fc
	}
	for fi := 0; fi < fc; fi++ {
		if fv[fi] < 0 || fv[fi] > 1 {
			return errors.New("feature value must be in range 0..1")
		}
		c.fs[base+fi] += fv[fi] * cv
		c.ft[fi] += fv[fi] * cv
	}
	return nil
}

func (c *Classifier) Predict(fv []float64) ([]float64, error) {
	p := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cp := 0.0                    // calculated class probability
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			if fv[fi] < 0 || fv[fi] > 1 {
				return nil, errors.New("feature value must be in range 0..1")
			}
			if c.ft[fi] == 0 {
				continue
			}
			fp := cf[fi] / c.ft[fi]
			cp = or(cp, and(fp, fv[fi]))
		}
		p[ci] = cp
	}
	return p, nil
}

func (c *Classifier) Detect(fv []float64) ([]float64, error) {
	p := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cp := 1.0                    // calculated class probability
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			if fv[fi] < 0 || fv[fi] > 1 {
				return nil, errors.New("feature value must be in range 0..1")
			}
			if c.ft[fi] == 0 {
				continue
			}
			fp := (cf[fi] / c.ft[fi])
			if fp == 0 {
				continue
			}
			fp = fp * fv[fi] // feature value limited by range 0 <= v <= 1, so we just reduce the probability proportionally
			cp = and(cp, fp) // all must match
		}
		p[ci] = cp
	}
	return p, nil
}

func (c *Classifier) Detect2(fv []float64) ([]float64, error) {
	p := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cp := 1.0                    // calculated class probability
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			if fv[fi] < 0 || fv[fi] > 1 {
				return nil, errors.New("feature value must be in range 0..1")
			}
			if c.ct[ci] == 0 {
				continue
			}
			fp := cf[fi] / c.ct[ci]
			cp = and(cp, xnor(fp, fv[fi]))
		}
		p[ci] = cp
	}
	return p, nil
}

func (c *Classifier) DetectRBF(fv []float64) ([]float64, error) {
	p := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cp := 0.0                    // calculated class probability
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			if fv[fi] < 0 || fv[fi] > 1 {
				return nil, errors.New("feature value must be in range 0..1")
			}
			if c.ct[ci] == 0 {
				continue
			}
			fp := cf[fi]/c.ct[ci] - fv[fi]
			cp += fp * fp
		}
		p[ci] = -math.Sqrt(cp)
	}
	return p, nil
}

func (c *Classifier) DetectRBF2(fv []float64) ([]float64, error) {
	p := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cp := 0.0                    // calculated class probability
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			if fv[fi] < 0 || fv[fi] > 1 {
				return nil, errors.New("feature value must be in range 0..1")
			}
			if c.ct[ci] == 0 {
				continue
			}
			fp := math.Abs(cf[fi]/c.ct[ci] - fv[fi])
			cp += fp * fp * fp
		}
		p[ci] = -cp
	}
	return p, nil
}

func (c *Classifier) Detect3(fv []float64) ([]float64, error) {
	score := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cv := 0.0                    // calculated class value
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			cv += (math.Log(cf[fi]) - math.Log(c.ftt[ci])) * fv[fi]
		}
		score[ci] = cv + math.Log(c.ct[ci]/c.tc)
	}
	return score, nil
}

func (c *Classifier) ClassProbs(class int) ([]float64, error) {
	if class < 0 || class >= c.cc {
		return nil, errors.New("unknown class")
	}
	cp := make([]float64, c.fc)
	base := class * c.fc
	copy(cp, c.fs[base:base+c.fc])
	for fi := 0; fi < c.fc; fi++ { // for each feature
		if c.ft[fi] > 0 {
			cp[fi] = (cp[fi] / c.ft[fi])
		}
	}
	return cp, nil
}

func (c *Classifier) FeatureProbs(feature int) ([]float64, error) {
	if feature < 0 || feature > c.fc {
		return nil, errors.New("unknown feature")
	}
	t := c.ft[feature]
	if t == 0 {
		return nil, nil
	}
	fp := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		fp[ci] = c.fs[base+feature] / t
	}
	return fp, nil
}
