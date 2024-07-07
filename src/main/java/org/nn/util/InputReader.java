package org.nn.util;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.nn.entity.ImageData;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;

public class InputReader {


    public Map<String,Object> readTrainedData(String resourcePath) throws IOException, URISyntaxException {
        URL path = this.getClass().getClassLoader().getResource(resourcePath);
        String content = new String(Files.readAllBytes(Paths.get(path.toURI())));
        ObjectMapper objectMapper = new ObjectMapper();
        var mnistArr = objectMapper.readValue(content, new TypeReference<Map<String,Object>>() {
        });

        return mnistArr;
    }

    public ImageData[] readImageData(String resourcePath) throws IOException, URISyntaxException {
        URL path = this.getClass().getClassLoader().getResource(resourcePath);
        String content = new String(Files.readAllBytes(Paths.get(path.toURI())));
        ObjectMapper objectMapper = new ObjectMapper();
        var mnistArr = objectMapper.readValue(content, new TypeReference<ImageData[]>() {
        });
        mnistArr = Arrays.asList(mnistArr).stream().map(imageData -> {
            imageData.setLabel(getOutputLabel(imageData.getLabelArr())); return imageData;}).toArray(ImageData[]::new);
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
