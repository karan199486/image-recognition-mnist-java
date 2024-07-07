package org.nn.service;

import lombok.RequiredArgsConstructor;
import org.nn.entity.*;
import org.nn.entity.core.Layer;
import org.nn.entity.core.NeuralNetwork;
import org.nn.entity.core.Neuron;
import org.nn.entity.core.Weight;
import org.nn.util.Util;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

@RequiredArgsConstructor
public class NNPredictionService {

    private final NeuralNetwork neuralNetwork;

    public void feedForward(double[] inputActivations) {
        List<Layer> layers = neuralNetwork.getLayers();
        layers.getFirst().setNeurons(
                Arrays.stream(inputActivations)
                        .mapToObj(inputActivation -> NeuronInitializer.getDefault().getWithValue(inputActivation))
                        .toList()
        );

        IntStream.range(1, layers.size()).forEach(layerI -> {
            var layer = layers.get(layerI);
            var prevLayer = layers.get(layerI-1);
            List<Neuron> neurons = layer.getNeurons();
            neurons.forEach(neuron -> {
                //calculate its activation
                var sum = neuron.getBias().getValue();
                List<Weight> weights = neuron.getWeights();
                for(int weightI = 0; weightI < weights.size(); weightI++) {
                    sum += (weights.get(weightI).getValue() * prevLayer.getNeurons().get(weightI).getValue());
                }
                neuron.setZ(sum);
                neuron.setValue(Util.calculateSigmoid(sum));
            });
        });
    }

    public List<Double> predict(double[] inputActivations) {
        feedForward(inputActivations);
        return neuralNetwork.getLayers().getLast().getNeurons()
                .stream().mapToDouble(Neuron::getValue).boxed().toList();
    }

    public double[] predictPrimitive(double[] inputActivations) {
        feedForward(inputActivations);
        return neuralNetwork.getLayers().getLast().getNeurons()
                .stream().mapToDouble(Neuron::getValue).toArray();
    }

    public double testAccuracy(MnistMatrix[] testDataArr) {
        var total = testDataArr.length;
        var success = 0;
        for(var testData : testDataArr) {
            double[] actualOutput = predictPrimitive(testData.dataLinear);
            int maxI = 0;
            double maxValue = actualOutput[maxI];
            for(var i = 1; i < actualOutput.length; i++) {
                if(actualOutput[i] > maxValue) {
                    maxI = i;
                    maxValue = actualOutput[i];
                }
            }

            if(maxI == testData.getLabel()) {
                success++;
            }
        }
        return (100 * success) / (double)total;
    }
}
