package com.umu.learning.ann;


import com.umu.learning.ann.ANN.ActivationFunction.ActivationFunction;
import com.umu.learning.ann.ANN.ActivationFunction.TanhActivationSpec;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.Network;
import com.umu.learning.ann.ANN.Tester;
import com.umu.learning.ann.ANN.Trainer;
import com.umu.learning.ann.Parser.FaceFacit;
import com.umu.learning.ann.Parser.FaceImage;
import com.umu.learning.ann.Parser.Parser;
import org.ejml.simple.SimpleMatrix;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Artificial Neural Network Program for evaluating face expressions in 20x20 maps
 * This program uses
 *  ejml-core-0.32
 *  ejml-ddense-0.32
 *  ejml-simple-0.32
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class Main {

    /**
     * Parameters used in the program. These parameters have been optimized.
     */
    public static final int INPUTS                             = 400; // Number of inputs to the network
    public static final int HIDDEN_LAYER_NODES                 = 9; // Number of hidden layer nodes in the hidden layer
    public static final int OUTPUTS                            = 4; // Number of outputs from the network
    public static final ActivationFunction ACTIVATION_FUNCTION = new TanhActivationSpec(); // Activation function used

    public static final double LEARNING_RATE        = 0.3; // Learning Rate
    public static final double TRAINING_PORTION     = 0.95; // Training portion, the rest is validation portion
    public static final double MIN_TRAINING_DELTA   = 0.005; // Value to determine when training converges
    public static final double MIN_VALIDATION_DELTA = 0.005; // Value to determine when validation converges
    public static final double MIN_ERROR_LIMIT      = 0.25; // Minimum of 75% correct
    public static final double EPOCH_LIMIT          = 100; // Max number of epochs until it gives up training


    /**
     * Generates a NxM weight matrix where N=outputs and M=inputs. Randomly sets the value of each element
     * within the interval [-0.25, 0.25].
     * @param outputs Number of outputs
     * @param inputs Number of inputs
     * @return A new randomly generated NxM weight matrix
     */
    private static SimpleMatrix generateWeightMatrix (int outputs, int inputs) {
        double[][] data = new double[outputs][inputs];
        for (int o = 0; o < outputs; o++) {
            for (int i = 0; i < inputs; i++) {
                Random rand = new Random();
                int n = rand.nextInt(100)-50;
                data[o][i] = (n/50.0)/4.0;
            }
        }
        return new SimpleMatrix(data);
    }

    /**
     * Main
     * @param args Containing three arguments in the following order: training, training facit and test.
     */
    public static void main(String[] args) {

        if (args.length < 3) {
            throw new IllegalArgumentException("How to run: program [training](.txt) [facit](.txt) [test](.txt)");
        }

        Parser parser = new Parser();

        try {

            // Load files from program arguments
            List<String> trainingFile = parser.loadFile(args[0]);
            List<String> facitFile    = parser.loadFile(args[1]);
            List<String> testFile     = parser.loadFile(args[2]);

            // Parse the data of each file
            List<FaceImage> trainingFaces = parser.parseData(trainingFile);
            List<FaceFacit> trainingFacit = parser.parseFacit(facitFile);
            List<FaceImage> testFaces     = parser.parseData(testFile);

            // Create weight matrices
            SimpleMatrix w1 = generateWeightMatrix(HIDDEN_LAYER_NODES, INPUTS);
            SimpleMatrix w2 = generateWeightMatrix(OUTPUTS, HIDDEN_LAYER_NODES);
            List<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
            weights.add(w1);
            weights.add(w2);

            // Generate list of input vectors
            List<ColumnVector> inputs = new ArrayList<ColumnVector>();
            for (FaceImage img : trainingFaces)
                inputs.add(img.data);

            // Generate list of output vectors
            List<ColumnVector> outputs = new ArrayList<ColumnVector>();
            for (FaceFacit facit : trainingFacit)
                outputs.add(facit.toOutputForm(OUTPUTS));

            // Generate list of performance test input vectors
            List<ColumnVector> testInputs = new ArrayList<ColumnVector>();
            for (FaceImage img : testFaces)
                testInputs.add(img.data);

            // Create network
            Network network = new Network(LEARNING_RATE, weights, ACTIVATION_FUNCTION);

            // Train network
            Trainer trainer = new Trainer();
            network = trainer.trainNetwork(network, inputs, outputs,
                    TRAINING_PORTION, MIN_TRAINING_DELTA, MIN_VALIDATION_DELTA, MIN_ERROR_LIMIT, EPOCH_LIMIT);

            // Test network
            Tester tester = new Tester();
            List<Integer> result = tester.testNetwork(network, testInputs);

            // Generate result file and print to console
            PrintWriter writer = new PrintWriter("result.txt", "UTF-8");
            int i = 1;
            for (Integer r : result) {
                writer.println("Image" + i + " " + r);
                System.out.println("Image" + i + " " + r);
                i++;
            }
            writer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
