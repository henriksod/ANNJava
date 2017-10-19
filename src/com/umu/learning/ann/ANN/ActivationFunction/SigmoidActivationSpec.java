package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * SigmoidActivationSpec is the sigmoid activation function
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
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
