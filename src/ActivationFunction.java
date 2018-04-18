

/**
 * ActivationFunction is an interface that all activation functions has to follow.
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public interface ActivationFunction {

    /**
     * The activation function takes a value x and returns the function output y.
     * @param x input
     * @return output
     */
    double function (double x);

    /**
     * The derivative of the activation function.
     * @param x input
     * @return output
     */
    double derivative (double x);

    /**
     * Description of the activation function.
     * @return description
     */
    String description ();
}
