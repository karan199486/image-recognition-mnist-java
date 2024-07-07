package org.nn.service;

import org.nn.entity.ImageData;
import org.nn.entity.core.*;
import org.nn.util.Util;

import java.time.Duration;
import java.util.List;
import java.util.stream.IntStream;


public class NNLearningService {

    private final NeuralNetwork neuralNetwork;
    private final NNPredictionService predictionService;
    private final int epoch = 30;
    private final double learningRate = 3;
    private final int batchSize = 10;

    public NNLearningService(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        this.predictionService = new NNPredictionService(neuralNetwork);
    }

    public void startLearn(ImageData[] mnistTrainData, ImageData[] mnistTestData) {
        //stochastic gradient descent
        IntStream.rangeClosed(1, epoch).forEach(i -> {

            var start = System.currentTimeMillis();

            int batches = Math.ceilDiv(mnistTrainData.length, batchSize);

            Util.shuffleArray(mnistTrainData);

            int startI = 0;
            for (int batch = 1; batch <= batches; batch++) {
                int endI = Math.min(startI + batchSize, mnistTrainData.length);
                for (int dataI = startI; dataI < endI; dataI++) {
                    ImageData trainData = mnistTrainData[dataI];
                    predictionService.feedForward(trainData.getDataArr());
                    backPropagate(trainData);
                }
                applyAvgGradientOverParameters();
                startI = endI;
            }
            double successPercent = predictionService.testAccuracy(mnistTestData);

            System.out.println("Test result after " + i + " iteration : " + successPercent + "% , total duration : "
                    + Util.printDuration(Duration.ofMillis(System.currentTimeMillis() - start)));
        });
    }

    private double getCost(ImageData trainData) {
        //run feedforward
        double[] actualOutput = predictionService.predictPrimitive(trainData.getDataArr());
        //obtain cost function
        return Util.getCost(actualOutput, trainData.getLabelArr());
    }

    private void backPropagate(ImageData trainData) {

        //find gradient of bias, weight, activation of output layer
        Layer currLayer = neuralNetwork.getLayers().getLast();
        Layer prevLayer = neuralNetwork.getLayers().get(neuralNetwork.getLayers().size() - 2);

        for (int i = currLayer.getNeurons().size() - 1; i >= 0; i--) {
            var neuron = currLayer.getNeurons().get(i);
            double aL = Util.getCostDerivative(neuron.getValue(), trainData.getLabelArr()[i]);
            neuron.addGradient(aL);
            double bL = aL * Util.getSigmoidDerivative(neuron.getZ());
            neuron.getBias().addGradient(bL);
            List<Weight> weights = neuron.getWeights();
            for (int j = 0; j < weights.size(); j++) {
                Neuron prevLN = prevLayer.getNeurons().get(j);
                Weight weight = weights.get(j);
                weight.addGradient(bL * prevLN.getValue());
            }
        }

        //back propagate to previous layers
        //find a'L = sum(wL+1 * b'L+1) , b'L = a'L * sig'(zL) , w'L = b'L * aL-1
        for (int i = neuralNetwork.getLayers().size() - 2; i >= 1; i--) {

            prevLayer = neuralNetwork.getLayers().get(i - 1);
            currLayer = neuralNetwork.getLayers().get(i);
            var nextLayer = neuralNetwork.getLayers().get(i + 1);

            for (int currLNI = 0; currLNI < currLayer.getNeurons().size(); currLNI++) {
                var currLN = currLayer.getNeurons().get(currLNI);
                double aL = 0;
                for (var nextLN : nextLayer.getNeurons()) {
                    var w = nextLN.getWeights().get(currLNI).getValue();
                    var bL = nextLN.getBias().getLastGradient();
                    aL += (w * bL);
                }
                currLN.addGradient(aL);

                double bL = aL * Util.getSigmoidDerivative(currLN.getZ());
                currLN.getBias().addGradient(bL);

                List<Weight> weights = currLN.getWeights();
                for (int j = 0; j < weights.size(); j++) {
                    Neuron prevLN = prevLayer.getNeurons().get(j);
                    Weight weight = weights.get(j);
                    weight.addGradient(bL * prevLN.getValue());
                }
            }
        }
    }

    private void applyAvgGradientOverParameters() {
        neuralNetwork.getLayers().stream().skip(1).forEach(layer -> {
            layer.getNeurons().forEach(neuron -> {
                var avgAL = neuron.getGradient() / batchSize;
                neuron.setValue(neuron.getValue() - learningRate * avgAL);
                neuron.setGradient(0); //reset gradient value

                Bias bias = neuron.getBias();
                var avgBL = bias.getGradient() / batchSize;
                bias.setValue(bias.getValue() - learningRate * avgBL);
                bias.setGradient(0); //reset gradient value
                bias.setLastGradient(0); //reset last gradient value

                neuron.getWeights().forEach(weight -> {
                    var avgWL = weight.getGradient() / batchSize;
                    weight.setValue(weight.getValue() - learningRate * avgWL);
                    weight.setGradient(0); //reset gradient value
                });
            });
        });
    }


}
