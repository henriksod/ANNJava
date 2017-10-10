package com.umu.learning.ann.ANN.MatrixUtils;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by Henrik on 10/10/2017.
 */
public class MatrixUtils {
    /**
     * Performs hadamard multiplication on two matrices, returns a new matrix.
     * @param a matrix a
     * @param b matrix b
     * @return result matrix
     */
    public static SimpleMatrix hadamard (SimpleMatrix a, SimpleMatrix b) {
        if (a.numCols() == b.numCols() || a.numRows() == b.numRows()) {
            SimpleMatrix c = new SimpleMatrix(a);
            for (int i = 0; i < a.numRows(); i++)
                for (int j = 0; j < a.numCols(); j++)
                    c.set(i, j, c.get(i,j) * b.get(i,j));
            return c;
        }
        throw new IllegalArgumentException(
                "Hadamard product is undefined for matrices with different dimensions\n" +
                "A = ("+a.numRows()+","+a.numCols()+") /= "+"("+b.numRows()+","+b.numCols()+") = B");
    }
}
