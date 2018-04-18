
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

/**
 * Trainer contains methods for training a Network.
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class Trainer {

    /**
     * Trains the network given various parameters, inputs and outputs.
     * Outputs the current epoch and the resulting error and deltas to console.
     * @param net Network to be trained
     * @param ins List of input vectors
     * @param outs List of output vectors
     * @param trainingPortion Training portion, the rest will be used for validation (error is based on validation set)
     * @param trainingDeltaLimit Value to decide when training converges
     * @param validationDeltaLimit Value to decide when validation converges
     * @param errorMaximum The highest allowed error on validation set
     * @param epochLimit Max number of epochs until it gives up training
     * @return Trained network
     */
    public Network trainNetwork(Network net,
                                List<ColumnVector> ins,
                                List<ColumnVector> outs,
                                double trainingPortion,
                                double trainingDeltaLimit,
                                double validationDeltaLimit,
                                double errorMaximum,
                                double epochLimit)
    {
        // Divide the inputs and outputs into portions
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
            // Compute previous results
            double prevValidation = validateEach(result, testSetIn, testSetOut);
            double prevTraining = validateEach(result, trainingSetIn, trainingSetOut);

            // Compute trained results and deltas
            Network training = trainEach(result, trainingSetIn, trainingSetOut);
            curValidation = validateEach(training, testSetIn, testSetOut);
            validationDelta = curValidation - prevValidation;
            trainingDelta = validateEach(training, trainingSetIn, trainingSetOut) - prevTraining;
            result = training;

            NumberFormat formatter = new DecimalFormat("#0.0000");

            // Print progress
            System.out.println("# Epoch "+epoch+"\terror\tvalidation\ttraining\t[Delta]");
            System.out.println("# \t\t"+formatter.format(curValidation)+
                               "\t"+formatter.format(validationDelta)+
                               "\t\t"+formatter.format(trainingDelta));

            epoch++;
            // End if training delta or validation delta has reached convergence limit but only if maximum error
            // has been reached or if epoch limit has been reached.
        } while ((Math.abs(trainingDelta) > trainingDeltaLimit && Math.abs(validationDelta) > validationDeltaLimit
                    || curValidation >= errorMaximum) && epoch < epochLimit);

        return result;
    }

    /**
     * Trains the network for each input output pair.
     * @param net Network to be trained
     * @param ins List of input vectors
     * @param outs List of output vectors
     * @return Trained network
     */
    private Network trainEach(Network net, List<ColumnVector> ins, List<ColumnVector> outs) {
        if (ins.size() == outs.size()) {
            for (int i = 0; i < ins.size(); i++) {
                List<PropagatedLayer> prls = net.propagateNet(ins.get(i));
                List<BackpropagatedLayer> bprls = net.backpropagateNet(outs.get(i), prls);
                net.updateNet(bprls);
            }
            return net;
        } else {
            throw new IndexOutOfBoundsException("Number of inputs not same as number of outputs!");
        }
    }

    /**
     * Validates the network for each input output pair.
     * @param net Network to be validated
     * @param ins List of input vectors
     * @param outs List of output vectors
     * @return Error which is the mean of each input output error
     */
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

    /**
     * Calculates the magnitude between two lists of doubles.
     * Accepts different sizes of the lists but will only evaluate the magnitude based on the smallest one.
     * @param x List x
     * @param y List y
     * @return Magnitude value which is the sum of absolute differences between all values in the lists.
     */
    private double magnitude (List<Double> x, List<Double> y) {
        double mag = 0;
        for (int i = 0, j = 0; i < x.size() && j < y.size(); i++, j++) {
            mag += Math.abs(x.get(i)-y.get(j));
        }

        return mag/Math.min(x.size(), y.size());
    }
}