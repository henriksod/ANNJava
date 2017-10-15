package com.umu.learning.ann.ANN.ActivationFunction;

/**
 * Created by Henrik on 10/10/2017.
 */
public interface ActivationFunction {

    /**
     * The activation function takes a value x and returns the function output y.
     * @param x input
     * @return output
     */
    double function (double x);

    /**
     * The derivative of the activation function.
     * @param x input
     * @return output
     */
    double derivative (double x);

    /**
     * Description of the activation function.
     * @return description
     */
    String description ();
}
