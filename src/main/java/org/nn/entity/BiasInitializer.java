package org.nn.entity;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.nn.entity.core.Bias;
import org.nn.util.Util;

@AllArgsConstructor
@NoArgsConstructor
public class BiasInitializer {

    private double valueFrom;
    private double valueTo;

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
