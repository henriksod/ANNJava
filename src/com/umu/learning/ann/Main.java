package com.umu.learning.ann;


import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import org.ejml.simple.SimpleMatrix;

public class Main {

    public static void main(String[] args) {
        double[][] mat1 = {{1.0,0.5},{0.3,0.7}};
        SimpleMatrix a = new SimpleMatrix(mat1);
        double[][] mat2 = {{2.0,2.0},{2.0,2.0}};
        SimpleMatrix b = new SimpleMatrix(mat2);
        SimpleMatrix c = MatrixUtils.hadamard(a,b);
        System.out.println(c.toString());
        System.out.println(":)");
    }
}
