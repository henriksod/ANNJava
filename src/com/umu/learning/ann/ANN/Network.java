package com.umu.learning.ann.ANN;

import com.umu.learning.ann.ANN.Layer.Layer;

import java.util.List;

/**
 * Created by Henrik on 10/12/2017.
 */
public class Network {
    List<Layer> layers;
    double learningRate;

    List<BackpropagatedLayer> backpropagateNet (ColumnVector target, List<PropagatedLayer> layers)
    {
        BackpropagateLayer layerL = layers.get(layers.size()-1).backpropagatedFinalLayer(layers.get(layers.size()-1), target);
        List<BackpropagateLayer> list = new List<>();
        list.add(layerL);

        for (int i = layers.get(layers.size()-2); i >= 0; i--)
        {
            BackpropagateLayer currentlayer = layers.get(i).backpropagate(layers.get(i), layerL);
            layerL = currentlayer;
            list.add(currentlayer);
        }

        Collections.reverse(list);
    }
}
