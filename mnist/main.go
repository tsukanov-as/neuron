package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/tsukanov-as/neuron"
)

// DRAFT

func main() {
	fmt.Println("2 layers (sensor -> or):")
	Mnist2LayersOR()
	fmt.Println("2 layers (sensor -> and):")
	Mnist2LayersAND()
	fmt.Println("2 layers (bayes):")
	Mnist2LayersBayes()
	fmt.Println("3 layers (sensor -> fixed and -> or):")
	Mnist3Layers()
	fmt.Println("2 layers (sensor -> rbf):")
	Mnist2LayersRBF()
	fmt.Println("3 layers (sensor -> adaptive and -> or):")
	fmt.Printf("wait a few minutes...\n\n")
	Mnist3Layers2()
}

func Mnist2LayersOR() {
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

	// tune
	for epoch := 0; epoch < 100; epoch++ {
		for _, r := range train {
			p, err := c.Predict(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) != argmax(p) {
				err = c.Learn(int(r[0]), r[1:])
				if err != nil {
					log.Fatal(err)
				}
			}
		}
		total_train := 0.0
		for _, r := range train {
			p, err := c.Predict(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total_train += 1
			}
		}
		total_test := 0.0
		for _, r := range test {
			p, err := c.Predict(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total_test += 1
			}
		}
		fmt.Printf("tune [epoch %d] accuracy on train: %f, on test: %f\n", epoch+1, total_train/float64(len(train)), total_test/float64(len(test)))
	}
}

func Mnist2LayersAND() {
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
		p, err := c.Detect2(r[1:])
		if err != nil {
			log.Fatal(err)
		}
		if int(r[0]) == argmax(p) {
			total += 1
		}
	}

	fmt.Println(total / float64(len(test)))
}

func Mnist2LayersBayes() {
	c := neuron.New(10, mnistImgLen)
	c.Init(1, 1)

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
		p, err := c.Detect3(r[1:])
		if err != nil {
			log.Fatal(err)
		}
		if int(r[0]) == argmax(p) {
			total += 1
		}
	}

	fmt.Println(total / float64(len(test)))

	// tune
	for epoch := 0; epoch < 100; epoch++ {
		for _, r := range train {
			p, err := c.Detect3(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) != argmax(p) {
				err = c.Learn(int(r[0]), r[1:])
				if err != nil {
					log.Fatal(err)
				}
			}
		}
		total_train := 0.0
		for _, r := range train {
			p, err := c.Detect3(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total_train += 1
			}
		}
		total_test := 0.0
		for _, r := range test {
			p, err := c.Detect3(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total_test += 1
			}
		}
		fmt.Printf("tune [epoch %d] accuracy on train: %f, on test: %f\n", epoch+1, total_train/float64(len(train)), total_test/float64(len(test)))
	}
}

func Mnist3Layers() {
	train, err := readMnistCsv("mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	createChan := func(d [][]float64) *neuron.Classifier {
		c := neuron.New(2, 4)
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
			([]float64{0, 1, 1, 0}),
			([]float64{1, 0, 0, 1}),
		}),
		createChan([][]float64{
			([]float64{1, 1, 0, 0}),
			([]float64{0, 0, 1, 1}),
		}),
		createChan([][]float64{
			([]float64{1, 0, 1, 0}),
			([]float64{0, 1, 0, 1}),
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

func Mnist2LayersRBF() {
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
		p, err := c.DetectRBF(r[1:])
		if err != nil {
			log.Fatal(err)
		}
		if int(r[0]) == argmax(p) {
			total += 1
		}
	}

	fmt.Println(total / float64(len(test)))

	// tune
	for epoch := 0; epoch < 100; epoch++ {
		for _, r := range train {
			p, err := c.DetectRBF(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) != argmax(p) {
				err = c.Learn(int(r[0]), r[1:])
				if err != nil {
					log.Fatal(err)
				}
			}
		}
		total_train := 0.0
		for _, r := range train {
			p, err := c.DetectRBF(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total_train += 1
			}
		}
		total_test := 0.0
		for _, r := range test {
			p, err := c.DetectRBF(r[1:])
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total_test += 1
			}
		}
		fmt.Printf("tune [epoch %d] accuracy on train: %f, on test: %f\n", epoch+1, total_train/float64(len(train)), total_test/float64(len(test)))
	}
}

// experimental
func Mnist3Layers2() {
	const N = 10000 // размеры тренировочной и тестовой выборок

	testMax := 0.0     // максимум точности в попытках на тестовой выборке
	trainingMax := 0.0 // соответствующая точность на тренировочной выборке
	tlim := 100        // количество попыток с разным рандомом

	for t := 0; t < tlim; t++ {
		fmt.Printf("attempt: %d\ncurrent time: %s\n", t, time.Now().String())

		rand.Seed(time.Now().UnixNano())

		dl := []*neuron.Classifier{
			neuron.New(10, mnistImgLen),
			neuron.New(10, mnistImgLen),
			neuron.New(10, mnistImgLen),
			neuron.New(10, mnistImgLen),
			neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
			// neuron.New(10, mnistImgLen),
		}

		train, err := readMnistCsv("mnist_train.csv")
		if err != nil {
			log.Fatal(err)
		}
		train = train[:N]

		// Рандом для AND нейронов

		vv := make([]float64, mnistImgLen)
		for _, d := range dl {
			d.Init(50, 0)
			for i := 0; i < 10; i++ {
				for j := range vv {
					vv[j] = rand.Float64()
				}
				err := d.Learn(i, vv)
				if err != nil {
					log.Fatal(err)
				}
			}
		}

		// Проходы по AND нейронам (по сути наивная кластеризация)

		st := make([]int, len(dl))
		for i := 0; i < 100; i++ {
			println(i)
			for _, r := range train {
				fv := r[1:]
				mx := 0.0
				mi := 0
				for di := range dl {
					pz, err := dl[di].Detect2(fv)
					if err != nil {
						log.Fatal(err)
					}
					if pz[int(r[0])] > mx {
						mx = pz[int(r[0])]
						mi = di
					}
				}
				// учим максимально отозвавшегося
				err = dl[mi].Learn(int(r[0]), fv)
				if err != nil {
					log.Fatal(err)
				}
				st[mi]++
			}
		}
		// fmt.Println(st)

		// Один проход для OR нейронов

		c := neuron.New(10, 10*len(dl))

		for _, r := range train {
			fv := r[1:]
			pp := make([]float64, 0, 10*len(dl))
			for _, d := range dl {
				pd, err := d.Detect2(fv)
				if err != nil {
					log.Fatal(err)
				}
				pp = append(pp, pd...)
			}

			err = c.Learn(int(r[0]), pp)
			if err != nil {
				log.Fatal(err)
			}
		}

		// Проверка на тестовой выборке.

		test, err := readMnistCsv("mnist_test.csv")
		if err != nil {
			log.Fatal(err)
		}

		test = test[:N]

		total := 0.0
		isMax := false

		for _, r := range test {
			fv := r[1:]
			pp := make([]float64, 0, 10*len(dl))
			for _, d := range dl {
				pd, err := d.Detect2(fv)
				if err != nil {
					log.Fatal(err)
				}
				pp = append(pp, pd...)
			}
			p, err := c.Predict(pp)
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total += 1
			}
		}

		if total/float64(len(test)) > testMax {
			testMax = total / float64(len(test))
			isMax = true
		}

		// Проверка на тренировочной выборке.

		total2 := 0.0

		for _, r := range train {
			fv := r[1:]
			pp := make([]float64, 0, 10*len(dl))
			for _, d := range dl {
				pd, err := d.Detect2(fv)
				if err != nil {
					log.Fatal(err)
				}
				pp = append(pp, pd...)
			}
			p, err := c.Predict(pp)
			if err != nil {
				log.Fatal(err)
			}
			if int(r[0]) == argmax(p) {
				total2 += 1
			}
		}

		if isMax {
			trainingMax = total2 / float64(len(train))
		}

		fmt.Printf("accuracy: test=%f, train=%f, total=%f\n\n", total/float64(len(test)), total2/float64(len(train)), (total+total2)/(float64(len(test))+float64(len(train))))
	}
	fmt.Println("max:", tlim, testMax, trainingMax)
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
				p, err := ch.Detect2(v[:])
				if err != nil {
					log.Fatal(err)
				}
				dst = append(dst, p...)
			}
		}
	}
	return dst
}
