package org.nn.entity.core;

import lombok.Data;
import org.nn.constant.LayerType;
import org.nn.entity.NeuronInitializer;

import java.util.List;
import java.util.stream.Stream;

@Data
public class Layer {

    private List<Neuron> neurons;
    private LayerType type;

    public Layer(LayerType type, int nCount, int nPrevCount) {
        this.type = type;
        NeuronInitializer neuronInitializer =
                new NeuronInitializer();
        neurons = Stream.generate(() -> {
            if(type != LayerType.Input)
                return neuronInitializer.getWithGaussianWeightBiases(nPrevCount);
            else
                return neuronInitializer.getEmpty();
        }).limit(nCount).toList();
    }

}
