package com.umu.learning.ann.ANN;

import com.umu.learning.ann.ANN.Layer.PropagatedLayer;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Tester contains methods for testing a Network
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class Tester {

    /**
     * Tests a network for performance, outputs the answer that the network guessed for each input vector.
     * @param net the network
     * @param ins list of input vectors
     * @return list of answers ranging from 1 to 4 (1=Happy, 2=Sad, 3=Mischievous, 4=Mad).
     */
    public List<Integer> testNetwork(Network net, List<ColumnVector> ins) {
        List<Integer> outs = new ArrayList<Integer>();
        for (int i = 0; i < ins.size(); i++) {
            List<PropagatedLayer> prls = net.propagateNet(ins.get(i));
            List<Double> out = MatrixUtils.matrixToList(prls.get(prls.size()-1).pOut);

            outs.add(1 + largest(out));
        }
        return outs;
    }

    /**
     * Finds largest value index in list of doubles.
     * @param list list of doubles
     * @return index with the largest value
     */
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
