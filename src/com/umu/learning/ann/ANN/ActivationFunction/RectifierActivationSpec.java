package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * Created by Henrik on 10/10/2017.
 */
public class RectifierActivationSpec implements ActivationFunction {
    @Override
    public double function(double x) {
        return Math.max(0,x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    @Override
    public String description() {
        return "ReLU";
    }
}
