package org.nn.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.NoArgsConstructor;

import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
@NoArgsConstructor
public class MnistMatrix {


    public double [][] data;
    @JsonProperty("input")
    public double[] dataLinear;

    private int nRows;
    private int nCols;

    @JsonProperty("output")
    public double[] labelMatrix;
    private int label;



    public MnistMatrix(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;
        data = new double[nRows][nCols];
        dataLinear = new double[nRows*nCols];
    }

    public double getValue(int r, int c) {
        return data[r][c];
    }

    public void setValue(int row, int col, double value) {
        data[row][col] = value;
        dataLinear[(row * nRows) + col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

}