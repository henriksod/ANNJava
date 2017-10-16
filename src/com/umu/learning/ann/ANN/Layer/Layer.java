package com.umu.learning.ann.ANN.Layer;

import com.umu.learning.ann.ANN.ActivationFunction.ActivationFunction;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import org.ejml.simple.SimpleMatrix;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Henrik on 10/11/2017.
 */
public class Layer {
    public SimpleMatrix lW;        // Weight matrix
    public ColumnVector lB;        // Bias vector
    public ActivationFunction lAS; // Activation Function

    /**
     * Propagates through one layer K, given the previous propagated layer J.
     * Calculates the output of layer K for both the activation function and it's derivative.
     * @param layerJ Layer to be propagated
     * @param layerK Previous, propagated layer
     * @return Propagated current layer
     */
    public PropagatedLayer propagate(PropagatedLayer layerJ, Layer layerK) {
        PropagatedLayer newPRL = new PropagatedLayer();

        SimpleMatrix w = layerK.lW;
        ColumnVector b = layerK.lB;
        ColumnVector x = layerJ.pOut;
        SimpleMatrix a = w.mult(x).plus(b);

        newPRL.lW = w;
        newPRL.lB = b;
        newPRL.pIn = x;
        newPRL.lAS = layerK.lAS;

        // Apply activation function to output
        List<Double> input = MatrixUtils.matrixToList(a);
        List<Double> foutput = input.stream().map(newPRL.lAS::function).collect(Collectors.toList()); //lAS - Activation layer
        List<Double> dfoutput = input.stream().map(newPRL.lAS::derivative).collect(Collectors.toList());

        newPRL.pOut = MatrixUtils.listToVector(foutput); //propagated layer
        newPRL.pDeriv = MatrixUtils.listToVector(dfoutput);

        return newPRL;
    }
}

