package main

import (
	"fmt"
	"log"

	"github.com/tsukanov-as/neuron"
)

// DRAFT

func main() {
	fmt.Println("2 layers:")
	Mnist2Layers()
	fmt.Println("3 layers:")
	Mnist3Layers()
}

func Mnist2Layers() {
	c := neuron.New(10, mnistImgLen)

	// Один просмотр тренировочной выборки.

	train, err := readMnistCsv("mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range train {
		err = c.Learn(int(r[0]), r[1:])
		if err != nil {
			log.Fatal(err)
		}
	}

	// Проверка на тестовой выборке.

	test, err := readMnistCsv("mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	total := 0.0

	for _, r := range test {
		p, err := c.Predict(r[1:])
		if err != nil {
			log.Fatal(err)
		}
		if int(r[0]) == argmax(p) {
			total += 1
		}
	}

	fmt.Println(total / float64(len(test)))
}

func Mnist3Layers() {
	train, err := readMnistCsv("mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	createChan := func(d [][]float64) *neuron.Classifier {
		c := neuron.New(2, 8)
		for cl, r := range d {
			err := c.Learn(cl, r)
			if err != nil {
				log.Fatal(err)
			}
		}
		return c
	}

	// Слой детекторов `AND`, в каждый пиксель смотрят 6 фиксированных детекторов в 3 каналах.
	// Исходные пиксели при этом продублированы в инвертированном варианте (нейрон NOT).
	L1Chans := [3]*neuron.Classifier{
		createChan([][]float64{
			complemented([]float64{0, 1, 1, 0}),
			complemented([]float64{1, 0, 0, 1}),
		}),
		createChan([][]float64{
			complemented([]float64{1, 1, 0, 0}),
			complemented([]float64{0, 0, 1, 1}),
		}),
		createChan([][]float64{
			complemented([]float64{1, 0, 1, 0}),
			complemented([]float64{0, 1, 0, 1}),
		}),
	}

	const featuresCount = (28 - 1) * (28 - 1) * (len(L1Chans) * 2)

	// Слой ассоциаторов `OR` поверх детекторов.
	L2 := neuron.New(10, featuresCount)

	// Один просмотр тренировочной выборки.

	for _, r := range train {
		src := r[1:]
		var farr [featuresCount]float64 // попытка удержать вектор на стеке
		fvec := detect(L1Chans[:], src, farr[0:0])
		err = L2.Learn(int(r[0]), fvec)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Проверка на тестовой выборке.

	test, err := readMnistCsv("mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	total := 0.0

	for _, r := range test {
		src := r[1:]
		var farr [featuresCount]float64 // попытка удержать вектор на стеке
		fvec := detect(L1Chans[:], src, farr[0:0])
		p, err := L2.Predict(fvec)
		if err != nil {
			log.Fatal(err)
		}
		if int(r[0]) == argmax(p) {
			total += 1
		}
	}

	fmt.Println(total / float64(len(test)))
}

// Цикл по всем элементам изображения 2x2 и детектирование
func detect(L1Chans []*neuron.Classifier, src, dst []float64) []float64 {
	for x := 0; x < 28-1; x++ {
		for y := 0; y < 28-1; y++ {
			var v [4]float64
			v[0] = src[(y+0)*28+(x+0)]
			v[1] = src[(y+0)*28+(x+1)]
			v[2] = src[(y+1)*28+(x+0)]
			v[3] = src[(y+1)*28+(x+1)]
			for _, ch := range L1Chans {
				p, err := ch.Detect(complemented(v[:]))
				if err != nil {
					log.Fatal(err)
				}
				dst = append(dst, p...)
			}
		}
	}
	return dst
}
