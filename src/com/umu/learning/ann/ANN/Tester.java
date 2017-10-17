package com.umu.learning.ann.ANN;

import com.umu.learning.ann.ANN.Layer.PropagatedLayer;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Henrik on 10/17/2017.
 */
public class Tester {

    public List<Integer> testNetwork(Network net, List<ColumnVector> ins) {
        List<Integer> outs = new ArrayList<Integer>();
        for (int i = 0; i < ins.size(); i++) {
            List<PropagatedLayer> prls = net.propagateNet(ins.get(i));
            List<Double> out = MatrixUtils.matrixToList(prls.get(prls.size()-1).pOut);

            outs.add(1 + largest(out));
        }
        return outs;
    }

    private int largest(List<Double> list) {
        int index = 0;
        for (int i = 0; i < list.size(); i++)
        {
            if ( list.get(i) > list.get(index) )
            {
                index = i;
            }
        }
        return index;
    }
}
