package org.nn.entity;

import lombok.RequiredArgsConstructor;
import org.nn.entity.core.Bias;
import org.nn.util.Util;

@RequiredArgsConstructor
public class BiasInitializer {

    private final double valueFrom;
    private final double valueTo;

    public Bias getRandom() {
        var bias = new Bias();
        bias.setValue(Util.getRandomValue(valueFrom, valueTo));
        return bias;
    }

    public Bias getGaussian() {
        var bias = new Bias();
        bias.setValue(Util.getRandomGaussianValue());
        return bias;
    }
}
