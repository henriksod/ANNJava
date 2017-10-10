package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * Created by Henrik on 10/10/2017.
 */
public class IdentityActivationSpec implements ActivationFunction {
    @Override
    public double function(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }

    @Override
    public String description() {
        return "identity";
    }
}
