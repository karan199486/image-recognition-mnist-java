package org.nn;

import org.nn.constant.LayerType;
import org.nn.entity.ImageData;
import org.nn.entity.core.Bias;
import org.nn.entity.core.NeuralNetwork;
import org.nn.entity.core.Weight;
import org.nn.util.InputReader;
import org.nn.service.NNLearningService;
import org.nn.service.NNPredictionService;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {

    //IMPORTANT : extract train_data.json, test_data.json, pretrained_data in resource folder from data.zip before running
    public static void main(String[] args) throws IOException, URISyntaxException {
        //create neural network with neuron count of each layer
        NeuralNetwork nn = new NeuralNetwork(new int[]{784, 30, 10});
        startLearning(nn);
//        testWithTrainedData();
    }

    private static void startLearning(NeuralNetwork nn) throws IOException, URISyntaxException {
        //prepare learning data
        InputReader inputReader = new InputReader();
        ImageData[] trainDataArr = inputReader.readImageData("train_data.json");
        ImageData[] testDataArr = inputReader.readImageData("test_data.json");
        //start learning
        var nnLearningService = new NNLearningService(nn);
        nnLearningService.startLearn(trainDataArr, testDataArr);
    }

    private static void testWithTrainedData() throws IOException, URISyntaxException {
        InputReader inputReader = new InputReader();
        Map<String, Object> data = inputReader.readTrainedData("pretrained_data_95per.json");
        ImageData[] testDataArr = inputReader.readImageData("test_data.json");

        NeuralNetwork nn = new NeuralNetwork(new int[]{784, 30, 10});
        nn.getLayers().forEach(layer -> {

            AtomicInteger i = new AtomicInteger(0);

            layer.getNeurons().forEach(neuron -> {
                int j = -1;
                if (layer.getType() == LayerType.Hidden) {
                    j = 0;
                } else if (layer.getType() == LayerType.Output) {
                    j = 1;
                } else {
                    return;
                }
                List<Double> weights = ((List<List<List<Double>>>) data.get("weight")).get(j).get(i.get());
                neuron.setWeights(weights.stream().map(w -> {
                    Weight weight = new Weight();
                    weight.setValue(w);
                    return weight;
                }).toList());
                Double bias = ((List<List<List<Double>>>) data.get("bias")).get(j).get(i.get()).get(0);
                Bias biasO = new Bias();
                biasO.setValue(bias);
                neuron.setBias(biasO);

                i.incrementAndGet();
            });
        });

        NNPredictionService nnPredictionService = new NNPredictionService(nn);
        double successPercent = nnPredictionService.testAccuracy(testDataArr);
        System.out.println("Test result : " + successPercent + "% success rate");
    }
}
