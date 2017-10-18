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
 * Created by Henrik on 10/12/2017.
 */
public class Network {
    List<Layer> layers = new ArrayList<Layer>();
    double learningRate;

    public Network (double learningRate, List<SimpleMatrix> weights, ActivationFunction aF) {
        if (weights.size() >= 2) {
            SimpleMatrix w0 = weights.get(0);
            List<SimpleMatrix> checkedWeights = new ArrayList<SimpleMatrix>();
            checkedWeights.add(w0);

            for (int i = 1; i < weights.size(); i++) {
                SimpleMatrix currentWeight = checkDimensions(w0,weights.get(i));
                w0 = currentWeight;
                checkedWeights.add(currentWeight);
            }

            for (SimpleMatrix w : checkedWeights) {
                Layer newLayer = buildLayer(w, aF);
                layers.add(newLayer);
            }

            this.learningRate = learningRate;
        } else {
            throw new IndexOutOfBoundsException("You need at least two weight matrices in the network!");
        }
    }

    public Network (double learningRate, List<SimpleMatrix> weights, ActivationFunction aF, ActivationFunction aFO) {
        this(learningRate,weights,aF);
        layers.get(layers.size()-1).lAS = aFO;
    }

    public List<PropagatedLayer> propagateNet (ColumnVector input)
    {
        PropagatedLayer layer0 = new PropagatedSensorLayer();
        layer0.pOut = validateInput(input);

        List<PropagatedLayer> list = new ArrayList<PropagatedLayer>();

        for (int i = 0; i < layers.size(); i++)
        {
            PropagatedLayer currentlayer = layers.get(i).propagate(layer0, layers.get(i));
            layer0 = currentlayer;
            list.add(currentlayer);
        }

        return list;
    }

    public List<BackpropagatedLayer> backpropagateNet (ColumnVector target, List<PropagatedLayer> layers)
    {
        BackpropagatedLayer layerL = layers.get(layers.size()-1).backpropagateFinalLayer(layers.get(layers.size()-1), target);
        List<BackpropagatedLayer> list = new ArrayList<BackpropagatedLayer>();
        list.add(layerL);

        for (int i = layers.size()-2; i >= 0; i--)
        {
            BackpropagatedLayer currentlayer = layers.get(i).backpropagate(layers.get(i), layerL);
            layerL = currentlayer;
            list.add(currentlayer);
        }

        Collections.reverse(list);
        return list;
    }

    public void updateNet (List<BackpropagatedLayer> bpLayers) {
        layers.clear();
        for (BackpropagatedLayer bpL : bpLayers) {
            layers.add(bpL.update(learningRate, bpL));
        }
    }

    private SimpleMatrix checkDimensions(SimpleMatrix a, SimpleMatrix b) {
        if (a.numRows() == b.numCols()) {
            return b;
        } else {
            throw new IndexOutOfBoundsException
                    ("Inconsistent dimensions in weight matrix");
        }
    }

    private ColumnVector validateInput(ColumnVector input) {
        SimpleMatrix firstLayerW = this.layers.get(0).lW;
        if (firstLayerW.numCols() == input.numRows()) {
            List<Double> inputs = MatrixUtils.matrixToList(input);
            boolean withinRange = inputs.stream().map(this::withinRange).reduce(true, (a,b) -> a && b);
            if (withinRange == true) {
                return input;
            } else {
                throw new NumberFormatException("Values in input vector not within range [0,1].");
            }
        } else {
            throw new IndexOutOfBoundsException
                    ("Number of rows in input vector doesn't match with number of columns in weight matrix.");
        }
    }

    private boolean withinRange(double value) {
        return value >= 0 && value <= 1;
    }

    private Layer buildLayer(SimpleMatrix w, ActivationFunction aF) {
        Layer newLayer = new Layer();
        newLayer.lW = w;
        newLayer.lAS = aF;

        double[] tmp = new double[w.numRows()];
        for (int i = 0; i < tmp.length; i++)
            tmp[i] = 1;

        newLayer.lB = new ColumnVector(tmp);

        return newLayer;
    }

}