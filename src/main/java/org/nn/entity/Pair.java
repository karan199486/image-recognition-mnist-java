package org.nn.entity;

import lombok.Data;

@Data
public class Pair <K,V>{
    private K key;
    private V value;
}
