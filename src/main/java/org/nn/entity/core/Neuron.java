package org.nn.entity.core;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class Neuron {
    private List<Weight> weights;
    private Bias bias;
    private double value;
    private double z;
    private List<Double> gradient = new ArrayList<>();

    public void addGradient(double value) {
        gradient.add(value);
    }
}
