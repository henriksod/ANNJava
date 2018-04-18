
import org.ejml.simple.SimpleMatrix;

import java.util.List;

/**
 * ColumnVector is an extension to SimpleMatrix that is strictly a Nx1 matrix.
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class ColumnVector extends SimpleMatrix {

    /**
     * First constructor of ColumnVector. Takes an array of double.
     * @param data array of double
     * @return
     */
    public ColumnVector(double[] data) {
        super(data.length, 1, false, data);
    }

    /**
     * Second constructor of ColumnVector. Takes a list of double.
     * @param data list of double
     * @return
     */
    public ColumnVector(List<Double> data) {
        super(data.size(), 1, false, listToPrimitive(data));
    }

    /**
     * Converts a SimpleMatrix class into a ColumnVector class. All values in the matrix
     * appears in one column in the column vector.
     * @param m matrix m
     * @return column vector
     */
    public static ColumnVector fromMatrix (SimpleMatrix m) {
        double[] data = new double[m.numRows()*m.numCols()];
        for (int i = 0; i < m.numRows(); i++)
            for (int j = 0; j < m.numCols(); j++)
                data[i+j*m.numRows()] = m.get(i,j);
        return new ColumnVector(data);
    }

    /**
     * Converts a list to it's primitive form.
     * @param data list of double
     * @return array of double
     */
    private static double[] listToPrimitive (List<Double> data) {
        double[] tmp = new double[data.size()];
        int idx = 0;
        for (Double d : data) {
            tmp[idx] = d; idx++;
        }
        return tmp;
    }

}