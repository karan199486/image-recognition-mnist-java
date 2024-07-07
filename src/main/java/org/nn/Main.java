package org.nn;

import org.nn.constant.LayerType;
import org.nn.entity.MnistMatrix;
import org.nn.entity.WeightInitializer;
import org.nn.entity.core.Bias;
import org.nn.entity.core.NeuralNetwork;
import org.nn.entity.core.Weight;
import org.nn.service.InputReader;
import org.nn.service.MnistDataReader;
import org.nn.service.NNLearningService;
import org.nn.service.NNPredictionService;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {

    public static void main(String[] args) throws IOException, URISyntaxException {
        InputReader inputReader = new InputReader();
//        var testLabelUrl = "/home/karanpanwar/Downloads/MNIST_ORG/t10k-labels.idx1-ubyte";
//        var testImageUrl = "/home/karanpanwar/Downloads/MNIST_ORG/t10k-images.idx3-ubyte";
//        var trainLabelUrl = "/home/karanpanwar/Downloads/MNIST_ORG/train-labels.idx1-ubyte";
//        var trainImageUrl = "/home/karanpanwar/Downloads/MNIST_ORG/train-images.idx3-ubyte";
//        MnistMatrix[] trainDataArr = inputReader.readMnistData(trainImageUrl, trainLabelUrl);
//        MnistMatrix[] testDataArr = inputReader.readMnistData(testImageUrl, testLabelUrl);
//
        MnistDataReader mnistDataReader = new MnistDataReader();
        MnistMatrix[] trainDataArr = mnistDataReader.readData("/home/karanpanwar/Downloads/MNIST_ORG/train_data.json");
        MnistMatrix[] testDataArr = mnistDataReader.readData("/home/karanpanwar/Downloads/MNIST_ORG/test_data.json");
        NeuralNetwork nn = new NeuralNetwork(new int[]{784, 30, 10});
        var nnLearningService = new NNLearningService(nn);
        nnLearningService.startLearn(trainDataArr, testDataArr);
//        testWithTrainedData(inputReader, testDataArr);
    }

    private static void testWithTrainedData(InputReader inputReader, MnistMatrix[] testDataArr) throws IOException {
        Map<String, Object> data = inputReader.readTrainedData("/home/karanpanwar/Downloads/neural-networks-and-deep-learning-master python3/neural-networks-and-deep-learning-master/src/wb.json");
        NeuralNetwork nn = new NeuralNetwork(new int[]{784, 30, 10});
        nn.getLayers().forEach(layer -> {

            AtomicInteger i  = new AtomicInteger(0);

            layer.getNeurons().forEach(neuron -> {
                int j = -1;
                if (layer.getType() == LayerType.Hidden) {
                    j = 0;
                } else if (layer.getType() == LayerType.Output) {
                    j = 1;
                } else {
                    return;
                }
                List<Double> weights = ((List<List<List<Double>>>)data.get("weight")).get(j).get(i.get());
                neuron.setWeights(weights.stream().map(w -> {
                    Weight weight = new Weight();
                    weight.setValue(w);
                    return weight;
                }).toList());
                Double bias = ((List<List<List<Double>>>)data.get("bias")).get(j).get(i.get()).get(0);
                Bias biasO = new Bias();
                biasO.setValue(bias);
                neuron.setBias(biasO);

                i.incrementAndGet();
            });
        });

        NNPredictionService nnPredictionService = new NNPredictionService(nn);
        nnPredictionService.testAccuracy(1, testDataArr);
    }
}
