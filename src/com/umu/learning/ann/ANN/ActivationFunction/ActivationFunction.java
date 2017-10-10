package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * Created by Henrik on 10/10/2017.
 */
public interface ActivationFunction {
    double function (double x);
    double derivative (double x);
    String description ();
}
