package com.umu.learning.ann.Parser;

import com.umu.learning.ann.ANN.MatrixUtils.ColumnMatrix;
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
 * Created by Henrik on 10/11/2017.
 */
public class Parser {
    private String imageNameRegex = "(Image(\\d+))";
    private String imageFacitRegex = "(Image(\\d+)\\s(\\d+))";
    private String imageContentRegex = "(\\d+)(\\s*\\d+)*";

    public List<FaceImage> parseData (List<String> lines) throws InterruptedException {
        ArrayList<FaceImage> faceList = new ArrayList<FaceImage>();
        Queue<String> queue = new Queue<String>();
        for (String s : lines)
            queue.enqueue(s);

        FaceImage currentFace = new FaceImage();
        int currentImage = 0;
        while (!queue.isEmpty()) {
            String line = queue.dequeue();
            Pattern p1 = Pattern.compile(imageNameRegex);
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
                        currentFace.data = new ColumnMatrix(numbers);
                    else
                        currentFace.data =
                                ColumnMatrix.fromMatrix(currentFace.data.concatRows(new ColumnMatrix(numbers)));
                }
            }
        }

        return faceList;
    }

    public List<FaceFacit> parseFacit (List<String> lines) throws InterruptedException {
        ArrayList<FaceFacit> faceList = new ArrayList<FaceFacit>();
        Queue<String> queue = new Queue<String>();
        for (String s : lines)
            queue.enqueue(s);

        FaceFacit currentFace = new FaceFacit();
        while (!queue.isEmpty()) {
            String line = queue.dequeue();
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

    public List<String> loadFile (String path) throws IOException {
        return Files.readAllLines(Paths.get(path), Charset.defaultCharset());
    }
}
