package com.umu.learning.ann.ANN;

import com.umu.learning.ann.ANN.ActivationFunction.ActivationFunction;
import com.umu.learning.ann.ANN.Layer.BackpropagatedLayer;
import com.umu.learning.ann.ANN.Layer.Layer;
import com.umu.learning.ann.ANN.Layer.PropagatedLayer;
import com.umu.learning.ann.ANN.Layer.PropagatedSensorLayer;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import org.ejml.data.Matrix;
import org.ejml.simple.SimpleMatrix;
import org.w3c.dom.ranges.RangeException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Network is the main ANN class which contains all the layers and
 * methods for propagating, backpropagating and updating.
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class Network {
    List<Layer> layers = new ArrayList<Layer>(); // A list of layers in the network (excluding input layer)
    double learningRate; // Learning rate of the network

    /**
     * Creates a new network object.
     * @param learningRate Learning rate
     * @param weights List of weight matrices. This defines the network and it's connections to each layer.
     * @param aF Activation function used
     * @return A new network
     */
    public Network (double learningRate, List<SimpleMatrix> weights, ActivationFunction aF) {
        if (weights.size() >= 1) {
            SimpleMatrix w0 = weights.get(0);
            List<SimpleMatrix> checkedWeights = new ArrayList<SimpleMatrix>();
            checkedWeights.add(w0);

            // Check weight matrix dimensions to the next and add to list
            for (int i = 1; i < weights.size(); i++) {
                SimpleMatrix currentWeight = checkDimensions(w0,weights.get(i));
                w0 = currentWeight;
                checkedWeights.add(currentWeight);
            }

            // Build the layers
            for (SimpleMatrix w : checkedWeights) {
                Layer newLayer = buildLayer(w, aF);
                layers.add(newLayer);
            }

            this.learningRate = learningRate;
        } else {
            throw new IndexOutOfBoundsException("You need at least one weight matrix to define a network!");
        }
    }

    /**
     * Creates a new network object with special activation function used for output layer.
     * @param learningRate Learning rate
     * @param weights List of weight matrices. This defines the network and it's connections to each layer.
     * @param aF Activation function used
     * @param aFO Activation function used for output layer
     * @return A new network
     */
    public Network (double learningRate, List<SimpleMatrix> weights, ActivationFunction aF, ActivationFunction aFO) {
        this(learningRate,weights,aF);
        layers.get(layers.size()-1).lAS = aFO;
    }

    /**
     * Propagates the network.
     * @param input input vector
     * @return A list of propagated layers with the last layer being the output layer.
     */
    public List<PropagatedLayer> propagateNet (ColumnVector input)
    {
        // Propagate input layer
        PropagatedLayer layer0 = new PropagatedSensorLayer();
        layer0.pOut = validateInput(input);

        List<PropagatedLayer> list = new ArrayList<PropagatedLayer>();

        // Propagate the rest
        for (int i = 0; i < layers.size(); i++)
        {
            PropagatedLayer currentlayer = layers.get(i).propagate(layer0, layers.get(i));
            layer0 = currentlayer;
            list.add(currentlayer);
        }

        return list;
    }

    /**
     * Backpropagates the network.
     * @param target target output vector
     * @param layers list of propagated layers
     * @return A list of backpropagated layers with the first layer being the input layer.
     */
    public List<BackpropagatedLayer> backpropagateNet (ColumnVector target, List<PropagatedLayer> layers)
    {
        // Backpropagate output layer
        BackpropagatedLayer layerL = layers.get(layers.size()-1).backpropagateFinalLayer(layers.get(layers.size()-1), target);
        List<BackpropagatedLayer> list = new ArrayList<BackpropagatedLayer>();
        list.add(layerL);

        // Backpropagate the rest
        for (int i = layers.size()-2; i >= 0; i--)
        {
            BackpropagatedLayer currentlayer = layers.get(i).backpropagate(layers.get(i), layerL);
            layerL = currentlayer;
            list.add(currentlayer);
        }

        Collections.reverse(list);
        return list;
    }

    /**
     * Updates the network
     * @param bpLayers list of backpropagated layers
     */
    public void updateNet (List<BackpropagatedLayer> bpLayers) {
        layers.clear();
        for (BackpropagatedLayer bpL : bpLayers) {
            layers.add(bpL.update(learningRate, bpL));
        }
    }

    /**
     * Checks dimensions of a matrix.
     * Throws an exception if number of rows in matrix A is not equal to number of columns in matrix B.
     * @param a matrix A
     * @param b matrix B
     * @return matrix B
     */
    private SimpleMatrix checkDimensions(SimpleMatrix a, SimpleMatrix b) {
        if (a.numRows() == b.numCols()) {
            return b;
        } else {
            throw new IndexOutOfBoundsException
                    ("Inconsistent dimensions in weight matrix");
        }
    }

    /**
     * Validates input vector.
     * Throws an exception if number of inputs is not equal tonumber of columns in first weight matrix.
     * Throws an exception if any value in the input vector is not within the range [0,1].
     * @param input input vector
     * @return input vector
     */
    private ColumnVector validateInput(ColumnVector input) {
        SimpleMatrix firstLayerW = this.layers.get(0).lW;
        if (firstLayerW.numCols() == input.numRows()) {
            List<Double> inputs = MatrixUtils.matrixToList(input);
            boolean withinRange = inputs.stream().map(this::withinRange).reduce(true, (a,b) -> a && b);
            if (withinRange) {
                return input;
            } else {
                throw new NumberFormatException("Values in input vector not within range [0,1].");
            }
        } else {
            throw new IndexOutOfBoundsException
                    ("Number of rows in input vector doesn't match with number of columns in weight matrix.");
        }
    }

    /**
     * Checks if a value is within range [0, 1].
     * @param value a double number
     * @return true if within range, otherwise false
     */
    private boolean withinRange(double value) {
        return value >= 0 && value <= 1;
    }

    /**
     * Builds a layer.
     * @param w weight matrix that connects this layer to the previous layer
     * @param aF activation function of this layer
     * @return new Layer
     */
    private Layer buildLayer(SimpleMatrix w, ActivationFunction aF) {
        Layer newLayer = new Layer();
        newLayer.lW = w;
        newLayer.lAS = aF;

        double[] tmp = new double[w.numRows()];
        for (int i = 0; i < tmp.length; i++)
            tmp[i] = 1;

        // Create bias vector, all elements are equal to 1 at the beginning
        newLayer.lB = new ColumnVector(tmp);

        return newLayer;
    }

}