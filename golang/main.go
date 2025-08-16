package main

import (
	"fmt"
	"math"
)

type Value struct {
	Data float64
	Grad float64

	backward func()

	parents []*Value

	label string
}

// Constructor
func NewValue(x float64, label string) *Value {
	out := &Value{
		Data:    x,
		label:   label,
		parents: []*Value{},
	}

	out.backward = func() {}

	return out
}

// Ops
func Add(a, b *Value) *Value {
	out := &Value{
		Data:    a.Data + b.Data,
		parents: []*Value{a, b},
		label:   fmt.Sprintf("(%s + %s)", a.label, b.label),
	}

	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}

	return out
}

func Mul(a, b *Value) *Value {
	out := &Value{
		Data:    a.Data * b.Data,
		parents: []*Value{a, b},
		label:   fmt.Sprintf("(%s * %s)", a.label, b.label),
	}

	out.backward = func() {
		a.Grad += out.Grad * b.Data
		b.Grad += out.Grad * a.Data
	}

	return out
}

func Tanh(a *Value) *Value {
	out := &Value{
		Data:    math.Tanh(a.Data),
		parents: []*Value{a},
		label:   fmt.Sprintf("tanh(%s)", a.label),
	}

	out.backward = func() {
		a.Grad += out.Grad * (1 - out.Data*out.Data)
	}

	return out
}

// Zero out the gradient of the node and all its parents to clear the previous backward pass
func (v *Value) ZeroGrad() {
	visited := map[*Value]bool{}
	var dfs func(v *Value)
	dfs = func(v *Value) {
		if visited[v] {
			return
		}
		v.Grad = 0
		visited[v] = true
		for _, parent := range v.parents {
			dfs(parent)
		}
	}
	dfs(v)
}

// Topological sort of the graph
func TopoSort(v *Value) []*Value {
	order := []*Value{}
	visited := map[*Value]bool{}
	var dfs func(v *Value)
	dfs = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, parent := range v.parents {
			dfs(parent)
		}
		order = append(order, v)
	}
	dfs(v)
	return order
}

// Backward pass of the graph
func (v *Value) Backward() {
	order := TopoSort(v)

	v.ZeroGrad()
	v.Grad = 1.0
	for i := len(order) - 1; i >= 0; i-- {
		order[i].backward()
	}
}

// Numerical gradient of a function
func numGrad(f func(float64) float64, x float64) float64 {
	eps := 1e-6
	return (f(x+eps) - f(x-eps)) / (2 * eps)
}

func main() {
	// build graph: x -> y=2x -> z=y+3 -> f=tanh(z)
	x := NewValue(1.0, "x")
	two := NewValue(2.0, "2")
	three := NewValue(3.0, "3")

	y := Mul(two, x)
	z := Add(y, three)
	f := Tanh(z)

	// seed gradient at the top and backprop
	f.Grad = 1.0
	f.Backward()

	// compare to numerical grad
	got := x.Grad
	want := numGrad(func(xx float64) float64 {
		return math.Tanh(2*xx + 3)
	}, 1.0)

	err := math.Abs(got - want)

	fmt.Printf("x.grad (backprop)=%.6f, (numerical)=%.6f, err=%.6f\n", got, want, err)
}
