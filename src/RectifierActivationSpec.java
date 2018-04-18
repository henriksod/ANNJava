
/**
 * RectifierActivationSpec is the ReLU activation function
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class RectifierActivationSpec implements ActivationFunction {
    @Override
    public double function(double x) {
        return Math.max(0,x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    @Override
    public String description() {
        return "ReLU";
    }
}
