package org.nn.entity.core;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class Bias {
    private double value;
    private List<Double> gradient = new ArrayList<>();

    public void addGradient(double value) {
        gradient.add(value);
    }
}
