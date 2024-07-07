package org.nn.entity;

import org.nn.entity.core.Neuron;

public class NeuronInitializer {

    private final BiasInitializer biasInitializer;
    private final WeightInitializer weightInitializer;

    public NeuronInitializer(BiasInitializer biasInitializer, WeightInitializer weightInitializer) {
        this.biasInitializer = biasInitializer;
        this.weightInitializer = weightInitializer;
    }

    public NeuronInitializer(double biasFrom, double biasTo, double weightFrom, double weightTo) {
        this.biasInitializer = new BiasInitializer(biasFrom, biasTo);
        this.weightInitializer = new WeightInitializer(weightFrom,weightTo);
    }

    public NeuronInitializer() {
        this.biasInitializer = new BiasInitializer(-100, 100);
        this.weightInitializer = new WeightInitializer(-100,100);
    }

    public static NeuronInitializer getDefault() {
        return new NeuronInitializer(-100, 100, -100, 100);
    }

    public Neuron getWithRandomWeightBiases(int nPrevCount) {
        Neuron neuron = new Neuron();
        neuron.setBias(biasInitializer.getRandom());
        neuron.setWeights(weightInitializer.getRandomList(nPrevCount));
        return neuron;
    }

    public Neuron getWithGaussianWeightBiases(int nPrevCount) {
        Neuron neuron = new Neuron();
        neuron.setBias(biasInitializer.getGaussian());
        neuron.setWeights(weightInitializer.getRandomGaussianList(nPrevCount));
        return neuron;
    }

    public Neuron getEmpty() {
        return new Neuron();
    }

    public Neuron getWithValue(double value) {
        var neuron = new Neuron();
        neuron.setValue(value);
        return neuron;
    }
}
