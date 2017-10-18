package com.umu.learning.ann.ANN.Layer;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by Henrik on 10/11/2017.
 */
public class BackpropagatedLayer extends Layer {
    public ColumnVector bpIn;          // Layer input vector
    public ColumnVector bpOut;         // Layer output vector
    public ColumnVector bpDazzle;      // Layer error signals
    public SimpleMatrix bpErrGrad;     // Layer weight error
    public ColumnVector bpBiasErrGrad; // Layer bias error
    public ColumnVector bpDeriv;       // Layer activation function derivative

    public Layer update (double lRate, BackpropagatedLayer bpLayer) {
        SimpleMatrix wOld = bpLayer.lW;
        ColumnVector bOld = bpLayer.lB;
        SimpleMatrix delW = bpLayer.bpErrGrad.scale(lRate);
        ColumnVector delB = ColumnVector.fromMatrix(bpLayer.bpBiasErrGrad.scale(lRate));
        SimpleMatrix wNew = wOld.minus(delW);
        ColumnVector bNew = ColumnVector.fromMatrix(bOld.minus(delB));

        Layer newLayer = new Layer();
        newLayer.lW = wNew;
        newLayer.lB = bNew;
        newLayer.lAS = bpLayer.lAS;

        return newLayer;
    }

}