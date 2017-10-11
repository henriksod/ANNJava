package com.umu.learning.ann.ANN.MatrixUtils;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by Henrik on 10/11/2017.
 */
public class ColumnMatrix extends SimpleMatrix {

    public ColumnMatrix(double[] data) {
        super(data.length, 1, false, data);
    }

    public static ColumnMatrix fromMatrix (SimpleMatrix m) {
        double[] data = new double[m.numRows()*m.numCols()];
        for (int i = 0; i < m.numRows(); i++)
            for (int j = 0; j < m.numCols(); j++)
                data[i+j*m.numRows()] = m.get(i,j);
        return new ColumnMatrix(data);
    }
}