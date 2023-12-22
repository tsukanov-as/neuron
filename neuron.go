package neuron

import (
	"errors"
)

type Classifier struct {
	cc int       // classes count
	fc int       // features count
	fs []float64 // feature statistics by class
	ft []float64 // features total
	ct []float64 // classes total
}

func New(classes, features int) *Classifier {
	c := &Classifier{
		cc: classes,
		fc: features,
		fs: make([]float64, classes*features),
		ft: make([]float64, features),
		ct: make([]float64, classes),
	}
	for i := range c.ct {
		c.ct[i] = 100 // experimental magic needed for Detect2
	}
	return c
}

func (c *Classifier) Learn(class int, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
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
	}
	return nil
}

// experimental
func (c *Classifier) Learn2(class int, cv float64, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
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

// analogue of naive Bayes classifier
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
			fp := (cf[fi] / c.ft[fi]) // probability by feature (how many occur in this class / total); we can calculate this in advance, but it will take more memory (+classes*features*float64)
			// fp := (cf[fi] / c.ft[fi]) * (cf[fi] / c.ct[ci]) // maybe so ...
			fp = fp * fv[fi] // feature value limited by range 0 <= v <= 1, so we just reduce the probability proportionally
			cp = or(cp, fp)
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
			fp := (cf[fi] / c.ct[ci])
			fp = imply(fp, fv[fi])
			cp = and(cp, fp) // all must match
		}
		p[ci] = cp
	}
	return p, nil
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
