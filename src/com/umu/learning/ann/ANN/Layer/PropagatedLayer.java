package com.umu.learning.ann.ANN.Layer;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by Henrik on 10/11/2017.
 */
public class PropagatedLayer extends Layer {
    public ColumnVector pIn;    // Layer input vector
    public ColumnVector pOut;   // Layer output vector
    public ColumnVector pDeriv; // Layer activation function derivative

    /**
     * Backpropagates one layer J, given the previous backpropagated layer K.
     * Calculates the error gradient from the output to the current layer through
     * the error gradients of the previous backpropagated layers. Also calculates
     * the bias error gradient for the bias points.
     * @param layerJ Layer to be backpropagated
     * @param layerK Previous, backpropagated layer
     * @return Backpropagated current layer
     */
    public BackpropagatedLayer backpropagate(PropagatedLayer layerJ, BackpropagatedLayer layerK) {
        BackpropagatedLayer newBPRL = new BackpropagatedLayer();

        SimpleMatrix wKT = layerK.lW.transpose();
        ColumnVector fAK = layerK.bpDeriv;
        ColumnVector fAJ = layerJ.pDeriv;
        ColumnVector dazzleK = layerK.bpDazzle;
        ColumnVector dazzleJ = (ColumnVector)wKT.mult(MatrixUtils.hadamard(dazzleK, fAK));

        //back propagated layer
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

    public BackpropagatedLayer backpropagateFinalLayer(PropagatedLayer layer, ColumnVector target)
    {
        BackpropagatedLayer newBPRL = new BackpropagatedLayer();
        ColumnVector fA = layer.pDeriv;
        ColumnVector dazzle = ColumnVector.fromMatrix(layer.pOut.minus(target));

        //back propagated layer
        newBPRL.bpDazzle = dazzle;
        newBPRL.bpErrGrad = errorGrad(dazzle, fA, layer.pIn);
        newBPRL.bpBiasErrGrad = errorGrad(dazzle, fA);
        newBPRL.bpDeriv = layer.pDeriv;
        newBPRL.bpIn = layer.pIn;
        newBPRL.bpOut = layer.pOut;
        newBPRL.lW = layer.lW;
        newBPRL.lB = layer.lB;
        newBPRL.lAS = layer.lAS;

        return newBPRL;
    }

    /**
     * Calculates the error gradient for weight matrices.
     * @param dazzle Error gradient for the prior layers
     * @param deriv Derivative of layer output
     * @param input Input to layer
     * @return Error gradient for weight matrix of current layer
     */
    private ColumnVector errorGrad (ColumnVector dazzle, ColumnVector deriv, ColumnVector input) {
        return (ColumnVector)MatrixUtils.hadamard(dazzle,deriv).mult(input);
    }

    /**
     * Calculates the error gradient for bias vectors.
     * @param dazzle Error gradient for the prior layers
     * @param deriv Derivative of layer output
     * @return Error gradient for bias vector of current layer
     */
    private ColumnVector errorGrad (ColumnVector dazzle, ColumnVector deriv) {
        return (ColumnVector)MatrixUtils.hadamard(dazzle,deriv);
    }

}
