package org.nn.entity.core;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nn.constant.LayerType;

import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
public class NeuralNetwork {

    private List<Layer> layers;

    public NeuralNetwork(int[] layerSizes) {

        if(layerSizes.length < 3) throw new IllegalArgumentException("A Neural Network cannot have layers less than 3");

        layers = new ArrayList<>();

        for(int i = 0; i < layerSizes.length; i++) {
            if(i == 0) {
                //setup input layer with empty neurons
                layers.add(new Layer(LayerType.Input, layerSizes[i], -1));
            } else if(i == layerSizes.length -1) {
                //setup output layer
                layers.add(new Layer(LayerType.Output, layerSizes[i], layerSizes[i-1]));
            } else {
                //setup hidden layer
                layers.add(new Layer(LayerType.Hidden, layerSizes[i], layerSizes[i-1]));
            }
        }
    }
}
