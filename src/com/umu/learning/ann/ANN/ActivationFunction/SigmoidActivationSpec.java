package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * Created by Henrik on 10/10/2017.
 */
public class SigmoidActivationSpec implements ActivationFunction {
    @Override
    public double function(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        return function(x) * (1 - function(x));
    }

    @Override
    public String description() {
        return "sigmoid";
    }
}
