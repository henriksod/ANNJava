package com.umu.learning.ann;


import com.umu.learning.ann.ANN.ActivationFunction.TanhActivationSpec;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;
import com.umu.learning.ann.ANN.Network;
import com.umu.learning.ann.ANN.Tester;
import com.umu.learning.ann.ANN.Trainer;
import com.umu.learning.ann.Parser.FaceFacit;
import com.umu.learning.ann.Parser.FaceImage;
import com.umu.learning.ann.Parser.Parser;
import org.ejml.simple.SimpleMatrix;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {

    public static final int INPUTS = 400; // Number of inputs to the network
    public static final int HIDDEN_LAYER_NODES = 9; // Number of hidden layer nodes in the hidden layer
    public static final int OUTPUTS = 4; // Number of outputs from the network

    public static final double LEARNING_RATE = 0.2; // Learning Rate
    public static final double TRAINING_PORTION = (4.0/5.0); // Training portion, the rest is validation portion
    public static final double MIN_TRAINING_DELTA = 0.005; // Value to determine when it converges
    public static final double MIN_VALIDATION_DELTA = 0.0005; // Value to determine when it converges
    public static final double MIN_ERROR_LIMIT = 0.25; // Minimum of 80% correct

    private static SimpleMatrix generateWeightMatrix (int outputs, int inputs) {
        double[][] data = new double[outputs][inputs];
        for (int o = 0; o < outputs; o++) {
            for (int i = 0; i < inputs; i++) {
                Random rand = new Random();
                int n = rand.nextInt(100)-50;
                data[o][i] = n/50.0;
            }
        }
        return new SimpleMatrix(data);
    }

    public static void main(String[] args) {

        if (args.length < 3) {
            throw new IllegalArgumentException("How to run: program [training](.txt) [facit](.txt) [test](.txt)");
        }

        Parser parser = new Parser();

        try {

            List<String> trainingFile = parser.loadFile(args[0]);
            List<String> facitFile    = parser.loadFile(args[1]);
            List<String> testFile     = parser.loadFile(args[2]);

            List<FaceImage> trainingFaces = parser.parseData(trainingFile);
            List<FaceFacit> trainingFacit = parser.parseFacit(facitFile);
            List<FaceImage> testFaces     = parser.parseData(testFile);

            SimpleMatrix w1 = generateWeightMatrix(HIDDEN_LAYER_NODES, INPUTS);
            SimpleMatrix w2 = generateWeightMatrix(OUTPUTS, HIDDEN_LAYER_NODES);
            List<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
            weights.add(w1);
            weights.add(w2);

            List<ColumnVector> inputs = new ArrayList<ColumnVector>();
            for (FaceImage img : trainingFaces)
                inputs.add(img.data);

            List<ColumnVector> outputs = new ArrayList<ColumnVector>();
            for (FaceFacit facit : trainingFacit)
                outputs.add(facit.toOutputForm(OUTPUTS));

            List<ColumnVector> testInputs = new ArrayList<ColumnVector>();
            for (FaceImage img : testFaces)
                testInputs.add(img.data);

            Network network = new Network(LEARNING_RATE, weights, new TanhActivationSpec());

            Trainer trainer = new Trainer();
            network = trainer.trainNetwork(network, inputs, outputs,
                    TRAINING_PORTION, MIN_TRAINING_DELTA, MIN_VALIDATION_DELTA, MIN_ERROR_LIMIT);

            Tester tester = new Tester();
            List<Integer> result = tester.testNetwork(network, testInputs);

            System.out.println("Result");
            for (Integer r : result)
                System.out.println("\t"+r);

            PrintWriter writer = new PrintWriter("result.txt", "UTF-8");
            int i = 1;
            for (Integer r : result)
                writer.println("Image"+(i++)+" "+r);
            writer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
