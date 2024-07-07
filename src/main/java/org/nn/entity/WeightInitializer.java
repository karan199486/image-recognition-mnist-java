package org.nn.entity;

import lombok.RequiredArgsConstructor;
import org.nn.entity.core.Weight;
import org.nn.util.Util;

import java.util.List;
import java.util.stream.Stream;

@RequiredArgsConstructor
public class WeightInitializer {

    private final double valueFrom;
    private final double valueTo;

    public Weight getRandom() {
        var w = Weight.builder();
        w.value(Util.getRandomValue(valueFrom, valueTo));
        return w.build();
    }

    public Weight getGaussian() {
        var w = Weight.builder();
        w.value(Util.getRandomGaussianValue());
        return w.build();
    }

    public List<Weight> getRandomGaussianList(int size) {
        return Stream.generate(this::getGaussian).limit(size).toList();
    }

    public List<Weight> getRandomList(int size) {
        return Stream.generate(this::getRandom).limit(size).toList();
    }
}
