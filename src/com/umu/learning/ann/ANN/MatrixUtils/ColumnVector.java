package com.umu.learning.ann.ANN.MatrixUtils;

import org.ejml.simple.SimpleMatrix;

import java.util.List;

/**
 * Created by Henrik on 10/11/2017.
 */
public class ColumnVector extends SimpleMatrix {

    public ColumnVector(double[] data) {
        super(data.length, 1, false, data);
    }

    public ColumnVector(List<Double> data) {
        super(data.size(), 1, false, listToPrimitive(data));
    }

    public static ColumnVector fromMatrix (SimpleMatrix m) {
        double[] data = new double[m.numRows()*m.numCols()];
        for (int i = 0; i < m.numRows(); i++)
            for (int j = 0; j < m.numCols(); j++)
                data[i+j*m.numRows()] = m.get(i,j);
        return new ColumnVector(data);
    }

    private static double[] listToPrimitive (List<Double> data) {
        double[] tmp = new double[data.size()];
        int idx = 0;
        for (Double d : data) {
            tmp[idx] = d; idx++;
        }
        return tmp;
    }

}