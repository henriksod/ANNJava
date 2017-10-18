package com.umu.learning.ann.ANN;

import com.umu.learning.ann.ANN.Layer.BackpropagatedLayer;
import com.umu.learning.ann.ANN.Layer.PropagatedLayer;
import com.umu.learning.ann.ANN.MatrixUtils.ColumnVector;
import com.umu.learning.ann.ANN.MatrixUtils.MatrixUtils;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

/**
 * Created by Henrik on 10/17/2017.
 */
public class Trainer {

    public Network trainNetwork(Network net,
                                List<ColumnVector> ins,
                                List<ColumnVector> outs,
                                double trainingPortion,
                                double trainingDeltaLimit,
                                double validationDeltaLimit,
                                double errorMinimum,
                                double epochLimit)
    {
        List<ColumnVector> trainingSetIn = ins.subList(0,(int)((ins.size()-1)*trainingPortion));
        List<ColumnVector> trainingSetOut = outs.subList(0,(int)((ins.size()-1)*trainingPortion));
        List<ColumnVector> testSetIn = ins.subList((int)((ins.size()-1)*trainingPortion)+1,ins.size()-1);
        List<ColumnVector> testSetOut = outs.subList((int)((ins.size()-1)*trainingPortion)+1,ins.size()-1);

        int epoch = 1;
        double trainingDelta = 0;
        double validationDelta = 0;
        double curValidation = 0;
        Network result = net;
        do {
            double prevValidation = validateEach(result, testSetIn, testSetOut);
            double prevTraining = validateEach(result, trainingSetIn, trainingSetOut);

            Network training = trainEach(result, trainingSetIn, trainingSetOut);
            curValidation = validateEach(training, testSetIn, testSetOut);
            validationDelta = curValidation - prevValidation;
            trainingDelta = validateEach(training, trainingSetIn, trainingSetOut) - prevTraining;
            result = training;

            NumberFormat formatter = new DecimalFormat("#0.0000");

            System.out.println("Epoch "+epoch+"\terror\tvalidation\ttraining\t[Delta]");
            System.out.println("\t\t\t"+formatter.format(curValidation)+
                               "\t"+formatter.format(validationDelta)+
                               "\t\t"+formatter.format(trainingDelta));

            epoch++;
        } while ((Math.abs(trainingDelta) > trainingDeltaLimit && Math.abs(validationDelta) > validationDeltaLimit
                    || curValidation >= errorMinimum) && epoch < epochLimit);

        return result;
    }

    private Network trainEach(Network net, List<ColumnVector> ins, List<ColumnVector> outs) {
        if (ins.size() == outs.size()) {
            Network training = net;
            for (int i = 0; i < ins.size(); i++) {
                List<PropagatedLayer> prls = training.propagateNet(ins.get(i));
                List<BackpropagatedLayer> bprls = training.backpropagateNet(outs.get(i), prls);
                training.updateNet(bprls);
            }
            return training;
        } else {
            throw new IndexOutOfBoundsException("Number of inputs not same as number of outputs!");
        }
    }

    private double validateEach(Network net, List<ColumnVector> ins, List<ColumnVector> outs) {
        if (ins.size() == outs.size()) {
            double magMean = 0;
            for (int i = 0; i < ins.size(); i++) {
                List<PropagatedLayer> prls = net.propagateNet(ins.get(i));
                magMean += magnitude(MatrixUtils.matrixToList(outs.get(i)),
                                     MatrixUtils.matrixToList(prls.get(prls.size()-1).pOut));
            }
            return magMean/ins.size();
        } else {
            throw new IndexOutOfBoundsException("Number of inputs not same as number of outputs!");
        }
    }

    private double magnitude (List<Double> x, List<Double> y) {
        double mag = 0;
        for (int i = 0, j = 0; i < x.size() && j < y.size(); i++, j++) {
            mag += Math.abs(x.get(i)-y.get(j));
        }

        return mag/Math.min(x.size(), y.size());
    }
}