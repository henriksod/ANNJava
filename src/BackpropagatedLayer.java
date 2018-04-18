
import org.ejml.simple.SimpleMatrix;

/**
 * BackpropagatedLayer is a backpropagated layer that allows for updating the weights.
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class BackpropagatedLayer extends Layer {
    public ColumnVector bpIn;          // Layer input vector
    public ColumnVector bpOut;         // Layer output vector
    public ColumnVector bpDazzle;      // Layer error signals
    public SimpleMatrix bpErrGrad;     // Layer weight error
    public ColumnVector bpBiasErrGrad; // Layer bias error
    public ColumnVector bpDeriv;       // Layer activation function derivative

    /**
     * Updates a backpropagated layer, given the learning rate.
     * This method updates the weight matrix of the given layer using the error gradient matrix computed
     * through recursive backpropagation.
     * @param lRate Learning rate
     * @param bpLayer Backpropagated layer
     * @return Updated layer
     */
    public Layer update (double lRate, BackpropagatedLayer bpLayer) {
        SimpleMatrix wOld = bpLayer.lW;
        ColumnVector bOld = bpLayer.lB;
        SimpleMatrix delW = bpLayer.bpErrGrad.scale(lRate);
        ColumnVector delB = ColumnVector.fromMatrix(bpLayer.bpBiasErrGrad.scale(lRate));
        SimpleMatrix wNew = wOld.minus(delW);
        ColumnVector bNew = ColumnVector.fromMatrix(bOld.minus(delB));

        Layer newLayer = new Layer();
        newLayer.lW = wNew;
        newLayer.lB = bNew;
        newLayer.lAS = bpLayer.lAS;

        return newLayer;
    }

}