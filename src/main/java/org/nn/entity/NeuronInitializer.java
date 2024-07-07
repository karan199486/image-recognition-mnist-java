package org.nn.entity;

import org.nn.entity.core.Neuron;

public class NeuronInitializer {

    private final BiasInitializer biasInitializer;
    private final WeightInitializer weightInitializer;

    public static final double DEFAULT_BIAS_MIN = -100;
    public static final double DEFAULT_BIAS_MAX = 100;
    public static final double DEFAULT_WEIGHT_MIN = -100;
    public static final double DEFAULT_WEIGHT_MAX = 100;

    public NeuronInitializer(BiasInitializer biasInitializer, WeightInitializer weightInitializer) {
        this.biasInitializer = biasInitializer;
        this.weightInitializer = weightInitializer;
    }

    public NeuronInitializer(double biasFrom, double biasTo, double weightFrom, double weightTo) {
        this.biasInitializer = new BiasInitializer(biasFrom, biasTo);
        this.weightInitializer = new WeightInitializer(weightFrom,weightTo);
    }

    public NeuronInitializer() {
        this.biasInitializer = new BiasInitializer(DEFAULT_BIAS_MIN, DEFAULT_BIAS_MAX);
        this.weightInitializer = new WeightInitializer(DEFAULT_WEIGHT_MIN,DEFAULT_WEIGHT_MAX);
    }

    public static NeuronInitializer getDefault() {
        return new NeuronInitializer();
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
