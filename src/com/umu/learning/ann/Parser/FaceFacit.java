package com.umu.learning.ann.Parser;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;

/**
 * FaceFacit class contains correct face expression
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class FaceFacit {
    public int id; // Image id
    public int answer; // Answer (ranging from 1 to 4)

    /**
     * Converts answer (1 to 4) to network output form e.g. 3 -> [0 0 1 0]'
     * @param numAnswers Number of answers available
     * @return Answer in network output form
     */
    public ColumnVector toOutputForm (int numAnswers) {
        double[] data = new double[numAnswers];
        for (int i = 0; i < data.length; i++)
            if (i == answer-1)
                data[i] = 1;
            else
                data[i] = 0;
        return new ColumnVector(data);
    }
}
