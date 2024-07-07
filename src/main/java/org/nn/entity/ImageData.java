package org.nn.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class ImageData {

    @JsonProperty("input")
    private double[] dataArr;

    @JsonProperty("output")
    private double[] labelArr;

    private int label;

}