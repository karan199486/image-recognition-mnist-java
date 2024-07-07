package org.nn.entity.core;

import lombok.Data;


@Data
public class Bias {
    private double value;
    private double gradient;
    private double lastGradient;

    public void addGradient(double value) {
        gradient += value;
        lastGradient = value;
    }
}
