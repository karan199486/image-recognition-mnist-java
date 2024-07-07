package org.nn.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.nn.entity.MnistMatrix;
import org.nn.util.Util;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MnistDataReader  {

//    public Map<Integer, double[]> labelMatrixMap;
//
//    public MnistDataReader() {
//        labelMatrixMap = new HashMap<>() {
//            {
//                put(0, getOutputLabelMatrix(0));
//                put(1, getOutputLabelMatrix(1));
//                put(2, getOutputLabelMatrix(2));
//                put(3, getOutputLabelMatrix(3));
//                put(4, getOutputLabelMatrix(4));
//                put(5, getOutputLabelMatrix(5));
//                put(6, getOutputLabelMatrix(6));
//                put(7, getOutputLabelMatrix(7));
//                put(8, getOutputLabelMatrix(8));
//                put(9, getOutputLabelMatrix(9));
//            }
//
//            private double[] getOutputLabelMatrix(int label) {
//                if(label < 0 || label > 9) throw new IllegalArgumentException("unsupported label found");
//                double[] result = new double[10];
//                result[label] = 1.0;
//                return result;
//            }
//        }


//    }

    public MnistMatrix[] readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        MnistMatrix[] data = new MnistMatrix[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {
            MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols);
            mnistMatrix.setLabel(labelInputStream.readUnsignedByte());
            mnistMatrix.labelMatrix = getOutputLabelMatrix(mnistMatrix.getLabel());
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    mnistMatrix.setValue(r, c, Util.convertBetween0And1(dataInputStream.readUnsignedByte(), 0, 255));
                }
            }
            data[i] = mnistMatrix;
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }

    public MnistMatrix[] readData(String dataFilePath) throws IOException {

        String content = new String(Files.readAllBytes(Paths.get(dataFilePath)));
        ObjectMapper objectMapper = new ObjectMapper();
        var mnistArr = objectMapper.readValue(content, new TypeReference<MnistMatrix[]>() {
        });
        mnistArr = Arrays.asList(mnistArr).stream().map(mnistMatrix -> {mnistMatrix.setLabel(getOutputLabel(mnistMatrix.labelMatrix)); return mnistMatrix;}).toArray(MnistMatrix[]::new);
        return mnistArr;
    }

    private double[] getOutputLabelMatrix(int label) {
        if(label < 0 || label > 9) throw new IllegalArgumentException("unsupported label found");
        double[] result = new double[10];
        result[label] = 1.0;
        return result;
    }

    private int getOutputLabel(double[] activations) {
        for(int i = 0; i < activations.length; i++)
            if(activations[i] == 1.0) return i;
        return -1;
    }
}