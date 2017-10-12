package com.umu.learning.ann.ANN.Layer;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;

/**
 * Created by Henrik on 10/11/2017.
 */
public class BackpropagatedLayer extends Layer {
    ColumnVector bpIn; // Layer input vector
    ColumnVector bpOut; // Layer output vector
    ColumnVector bpDazzle; // Layer error signals
    ColumnVector bpErrGrad; // Layer weight error
    ColumnVector bpBiasErrGrad; // Layer bias error
    ColumnVector bpDeriv; // Layer activation function derivative
}