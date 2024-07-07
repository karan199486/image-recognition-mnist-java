package org.nn.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.razorvine.pickle.Pickler;
import net.razorvine.pickle.Unpickler;
import org.nn.entity.MnistMatrix;
import org.nn.entity.NNInput;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Scanner;

public class InputReader {

    private MnistDataReader mnistDataReader = new MnistDataReader();

    public MnistMatrix[] readMnistData(String imagesUrl, String labelsUrl) throws URISyntaxException, IOException {
        MnistMatrix[] mnistMatrices = mnistDataReader.readData(imagesUrl, labelsUrl);
        return mnistMatrices;
    }
    //not working
//    public NNInput getInputFromPickleFile(String filePath) throws IOException, URISyntaxException {
//        byte[] bytes = Files.readAllBytes(Path.of(new URI("file://"+filePath)));
//        Unpickler.registerConstructor("numpy","core.multiarray");
//        Object data = new Unpickler().loads(bytes);
//        System.out.println(data);
//        return null;
//    }

    public Map<String,Object> readTrainedData(String dataPath) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(dataPath)));
        ObjectMapper objectMapper = new ObjectMapper();
        var mnistArr = objectMapper.readValue(content, new TypeReference<Map<String,Object>>() {
        });

        return mnistArr;
    }
}
