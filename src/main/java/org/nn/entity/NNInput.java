package org.nn.entity;

import lombok.Data;

@Data
public class NNInput {
    private Pair<double[][], double[]> data;
}
