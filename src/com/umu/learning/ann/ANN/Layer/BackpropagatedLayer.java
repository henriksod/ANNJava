package com.umu.learning.ann.ANN.Layer;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;

/**
 * Created by Henrik on 10/11/2017.
 */
public class BackpropagatedLayer extends Layer {
    public ColumnVector bpIn;          // Layer input vector
    public ColumnVector bpOut;         // Layer output vector
    public ColumnVector bpDazzle;      // Layer error signals
    public ColumnVector bpErrGrad;     // Layer weight error
    public ColumnVector bpBiasErrGrad; // Layer bias error
    public ColumnVector bpDeriv;       // Layer activation function derivative
}