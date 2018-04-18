
import sun.misc.Queue;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Parser class loads and parses files containing faces and correct answers
 *
 * @author HenrikS 2017-10-10
 * @author JoanaV 2017-10-10
 */
public class Parser {

    /**
     * Parses a file using regular expressions to get image data of a 20x20 image contained in a text file.
     * Each image should have the following form:
     *      ImageX
     *      d1x1 d2x1 d3x1 ... d20x1
     *      d1x2 d2x2 d3x2 ... d20x2
     *      .
     *      .
     *      .
     *      d1x20 d2x20 d3x20 ... d20x20
     * @param lines lines of the contents of the file
     * @return list of FaceImage
     */
    public List<FaceImage> parseData (List<String> lines) throws InterruptedException {
        ArrayList<FaceImage> faceList = new ArrayList<FaceImage>();
        Queue<String> queue = new Queue<String>();
        for (String s : lines)
            queue.enqueue(s);

        FaceImage currentFace = new FaceImage();
        int currentImage = 0;
        while (!queue.isEmpty()) {
            String line = queue.dequeue();
            String imageNameRegex = "(Image(\\d+))";
            Pattern p1 = Pattern.compile(imageNameRegex);
            String imageContentRegex = "(\\d+)(\\s*\\d+)*";
            Pattern p2 = Pattern.compile(imageContentRegex);
            Matcher nameMatcher = p1.matcher(line);
            Matcher contentMatcher = p2.matcher(line);

            if (!line.startsWith("#")) {
                if (nameMatcher.matches()) {
                    currentImage = Integer.parseInt(nameMatcher.group(2));
                    currentFace = new FaceImage();
                    currentFace.id = currentImage;
                    faceList.add(currentFace);
                } else if (contentMatcher.matches()) {
                    String data = contentMatcher.group(0);
                    String[] values = data.split(" ");
                    double[] numbers = new double[values.length];
                    for (int i = 0; i < values.length; i++)
                        numbers[i] = Integer.parseInt(values[i])/32.0;

                    if (currentFace.data == null)
                        currentFace.data = new ColumnVector(numbers);
                    else
                        currentFace.data =
                                ColumnVector.fromMatrix(currentFace.data.concatRows(new ColumnVector(numbers)));
                }
            }
        }

        return faceList;
    }

    /**
     * Parses a file using regular expressions to get image facit value of a specific image,
     * contained in a text file. Each image facit should have the following form where Y = [1,2,3,4]:
     *      ImageX Y
     * @param lines lines of the contents of the file
     * @return list of FaceFacit
     */
    public List<FaceFacit> parseFacit (List<String> lines) throws InterruptedException {
        ArrayList<FaceFacit> faceList = new ArrayList<FaceFacit>();
        Queue<String> queue = new Queue<String>();
        for (String s : lines)
            queue.enqueue(s);

        FaceFacit currentFace = new FaceFacit();
        while (!queue.isEmpty()) {
            String line = queue.dequeue();
            String imageFacitRegex = "(Image(\\d+)\\s(\\d+))";
            Pattern p = Pattern.compile(imageFacitRegex);
            Matcher nameMatcher = p.matcher(line);

            if (!line.startsWith("#")) {
                if (nameMatcher.matches()) {
                    int id = Integer.parseInt(nameMatcher.group(2));
                    int answer = Integer.parseInt(nameMatcher.group(3));
                    currentFace = new FaceFacit();
                    currentFace.id = id;
                    currentFace.answer = answer;
                    faceList.add(currentFace);
                }
            }
        }

        return faceList;
    }

    /**
     * Loads a file and returns a list of lines.
     * @param path file path
     * @return list of lines
     */
    public List<String> loadFile (String path) throws IOException {
        return Files.readAllLines(Paths.get(path), Charset.defaultCharset());
    }
}
