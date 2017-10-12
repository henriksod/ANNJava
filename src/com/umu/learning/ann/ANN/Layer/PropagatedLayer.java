package com.umu.learning.ann.ANN.Layer;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by Henrik on 10/11/2017.
 */
public class PropagatedLayer extends Layer {
    ColumnVector pIn; // Layer input vector
    ColumnVector pOut; // Layer output vector
    ColumnVector pDeriv; // Layer activation function derivative

    BackpropagatedLayer backpropagate(PropagatedLayer layerJ, BackpropagatedLayer layerK) {
        BackpropagatedLayer newBPRL = new BackpropagatedLayer();

        SimpleMatrix wKT = layerK.lW.transpose();
        ColumnVector fAK = layerK.bpDeriv;
        ColumnVector fAJ = layerJ.pDeriv;
        ColumnVector dazzleK = layerK.bpDazzle;
        ColumnVector dazzleJ = (ColumnVector)wKT.mult(MatrixUtils.hadamard(dazzleK, fAK));

        newBPRL.bpDazzle = dazzleJ;
        newBPRL.bpErrGrad = errorGrad(dazzleJ, fAJ, layerJ.pIn);
        newBPRL.bpBiasErrGrad = errorGrad(dazzleJ, fAJ);
        newBPRL.bpDeriv = layerJ.pDeriv;
        newBPRL.bpIn = layerJ.pIn;
        newBPRL.bpOut = layerJ.pOut;
        newBPRL.lW = layerJ.lW;
        newBPRL.lB = layerJ.lB;
        newBPRL.lAS = layerJ.lAS;

        return newBPRL;
    }

    private ColumnVector errorGrad (ColumnVector dazzle, ColumnVector deriv, ColumnVector input) {
        return (ColumnVector)MatrixUtils.hadamard(dazzle,deriv).mult(input);
    }

    private ColumnVector errorGrad (ColumnVector dazzle, ColumnVector deriv) {
        return (ColumnVector)MatrixUtils.hadamard(dazzle,deriv);
    }

}
