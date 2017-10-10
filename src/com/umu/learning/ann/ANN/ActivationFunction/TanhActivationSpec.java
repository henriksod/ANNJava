package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * Created by Henrik on 10/10/2017.
 */
public class TanhActivationSpec implements ActivationFunction {
    @Override
    public double function(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        return 1.0-Math.pow(function(x),2);
    }

    @Override
    public String description() {
        return "tanh";
    }
}
