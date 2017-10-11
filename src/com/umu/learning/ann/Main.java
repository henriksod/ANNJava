package com.umu.learning.ann;


import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import com.umu.learning.ann.Parser.FaceFacit;
import com.umu.learning.ann.Parser.FaceImage;
import com.umu.learning.ann.Parser.Parser;
import org.ejml.simple.SimpleMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        double[][] mat1 = {{1.0,0.5},{0.3,0.7}};
        SimpleMatrix a = new SimpleMatrix(mat1);
        double[][] mat2 = {{2.0,2.0},{2.0,2.0}};
        SimpleMatrix b = new SimpleMatrix(mat2);
        SimpleMatrix c = MatrixUtils.hadamard(a,b);
        System.out.println(c.toString());
        System.out.println(":)");

        Parser parser = new Parser();
        try {
            List<String> lines =
                    parser.loadFile("C:\\Users\\Henrik\\IdeaProjects\\ANNJava\\input\\training.txt");
            List<String> linesFacit =
                    parser.loadFile("C:\\Users\\Henrik\\IdeaProjects\\ANNJava\\input\\training-facit.txt");
            try {
                List<FaceImage> faces = parser.parseData(lines);
                List<FaceFacit> facit = parser.parseFacit(linesFacit);
                System.out.println("Data");
                System.out.println("Image"+faces.get(0).id);
                System.out.println(faces.get(0).data);
                System.out.println("Facit");
                System.out.println("Image"+facit.get(0).id+"\t"+facit.get(0).answer);
            } catch (InterruptedException e2) {
                e2.printStackTrace();
            }
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }
}
