package neuron

import "errors"

type Classifier struct {
	cc int       // classes count
	fc int       // features count
	fs []float64 // feature statistics by class
	ft []float64 // features total
}

func New(classes, features int) *Classifier {
	return &Classifier{
		cc: classes,
		fc: features,
		fs: make([]float64, classes*features),
		ft: make([]float64, features),
	}
}

func (c *Classifier) Learn(class int, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
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
			fp := cf[fi] / c.ft[fi] // probability by feature (how many occur in this class / total); we can calculate this in advance, but it will take more memory (+classes*features*float64)
			fp = fp * fv[fi]        // feature value limited by range 0 <= v <= 1, so we just reduce the probability proportionally
			cp = cp + fp - cp*fp    // addition theorem of probability (equivalent of logical OR): P(A+B) = P(A) + P(B) - P(A*B)
		}
		p[ci] = cp
	}
	return p, nil
}
