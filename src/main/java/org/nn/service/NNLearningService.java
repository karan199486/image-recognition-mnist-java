package org.nn.service;

import org.nn.entity.MnistMatrix;
import org.nn.entity.core.*;
import org.nn.util.Util;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;


public class NNLearningService {

    private final NeuralNetwork neuralNetwork;

    private final NNPredictionService predictionService;

    private InputReader inputReader;

    int epoch = 30;
    double learningRate = 3;
    int batchSize = 10;

    public NNLearningService(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        this.predictionService = new NNPredictionService(neuralNetwork);
    }

    public void startLearn(MnistMatrix[] mnistTrainData, MnistMatrix[] mnistTestData) {
        predictionService.testAccuracy(-1, mnistTestData);
        //stochastic gradient descent
        IntStream.rangeClosed(1, epoch).forEach(i -> {

            System.out.println("Started epoch " + i);

            int batches = Math.ceilDiv(mnistTrainData.length, batchSize);

            Util.shuffleArray(mnistTrainData);

            int startI = 0;
            for(int batch = 1; batch <= batches; batch++) {
                int endI = Math.min(startI + batchSize, mnistTrainData.length);
                for(int dataI = startI; dataI < endI; dataI++) {
                    MnistMatrix trainData = mnistTrainData[dataI];
                    predictionService.feedForward(trainData.dataLinear);
                    backPropagate(trainData);
                }
                applyAvgGradientOverParameters();
                startI = endI;
            }
            predictionService.testAccuracy(i, mnistTestData);
        });
    }

    private double getCost(MnistMatrix trainData) {
        //run feedforward
        double[] actualOutput = predictionService.predictPrimitive(trainData.dataLinear);
        //obtain cost function
        return Util.getCost(trainData.labelMatrix, actualOutput);
    }

    private void backPropagate(MnistMatrix trainData) {

        //find gradient of bias, weight, activation of output layer
        Layer currLayer = neuralNetwork.getLayers().getLast();
        Layer prevLayer = neuralNetwork.getLayers().get(neuralNetwork.getLayers().size()-2);

        for(int i = currLayer.getNeurons().size()-1; i >=0; i--) {
            var neuron = currLayer.getNeurons().get(i);
            double aL = Util.getCostDerivative(neuron.getValue(), trainData.labelMatrix[i]);
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
        for(int i = neuralNetwork.getLayers().size()-2; i >= 1; i--) {

            prevLayer = neuralNetwork.getLayers().get(i-1);
            currLayer = neuralNetwork.getLayers().get(i);
            var nextLayer = neuralNetwork.getLayers().get(i+1);

            for(int currLNI = 0; currLNI < currLayer.getNeurons().size(); currLNI++) {
                var currLN = currLayer.getNeurons().get(currLNI);
                double aL = 0;
                for(var nextLN : nextLayer.getNeurons()) {
                    var w = nextLN.getWeights().get(currLNI).getValue();
                    var bL = nextLN.getBias().getGradient().getLast();
                    aL += (w*bL);
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
                var avgAL = neuron.getGradient().stream().mapToDouble(d -> d).average().getAsDouble();
                neuron.setValue(neuron.getValue() - learningRate * avgAL);
                neuron.setGradient(new ArrayList<>()); //reset

                Bias bias = neuron.getBias();
                var avgBL = bias.getGradient().stream().mapToDouble(d->d).average().getAsDouble();
                bias.setValue(bias.getValue() - learningRate * avgBL);
                bias.setGradient(new ArrayList<>()); //reset

                neuron.getWeights().forEach(weight -> {
                    var avgWL = weight.getGradient().stream().mapToDouble(d->d).average().getAsDouble();
                    weight.setValue(weight.getValue() - learningRate * avgWL);
                    weight.setGradient(new ArrayList<>()); //reset
                });
            });
        });
    }


}
