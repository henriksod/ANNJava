package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * IdentityActivationSpec is the identity activation function
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
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
