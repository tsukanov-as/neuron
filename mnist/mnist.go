package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
)

const (
	mnistImgLen = 28 * 28
	mnistRecLen = 1 + mnistImgLen
)

func readMnistCsv(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal("Unable to read input file "+path, err)
	}
	defer f.Close()

	var mnist [][]float64

	rd := csv.NewReader(f)
	for {
		r, err := rd.Read()
		if r == nil || err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(r) != mnistRecLen {
			return nil, fmt.Errorf("unxepected record len: %d", len(r))
		}
		vec := make([]float64, mnistRecLen)
		mnist = append(mnist, vec)
		cl, err := strconv.Atoi(r[0])
		if err != nil {
			return nil, err
		}
		vec[0] = float64(cl)
		for i := 1; i < mnistRecLen; i++ {
			if r[i] != "0" {
				v, err := strconv.Atoi(r[i])
				if err != nil {
					return nil, err
				}
				vec[i] = float64(v) / 255
			}
		}
	}
	return mnist, nil
}

func printImg(img []float64) {
	for i := 0; i < mnistImgLen; i++ {
		if img[i] > 0 {
			print("#")
		} else {
			print(" ")
		}
		if i%28 == 0 {
			println()
		}
	}
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

func complemented(x []float64) []float64 {
	lim := len(x)
	for i := 0; i < lim; i++ {
		x = append(x, 1-x[i])
	}
	return x
}
