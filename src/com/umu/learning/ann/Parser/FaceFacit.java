package com.umu.learning.ann.Parser;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;

/**
 * Created by Henrik on 10/11/2017.
 */
public class FaceFacit {
    public int id;
    public int answer;

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
