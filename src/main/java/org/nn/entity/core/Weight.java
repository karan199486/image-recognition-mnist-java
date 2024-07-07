package org.nn.entity.core;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Weight {
//    private int fromIndex;
//    private int toIndex;
    private double value;
    private double gradient;

    public void addGradient(double value) {
        gradient += value;
    }
}
