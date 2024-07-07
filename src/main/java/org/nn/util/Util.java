package org.nn.util;

import java.time.Duration;
import java.util.Random;
import java.util.stream.IntStream;

public class Util {

    private Util(){}

    private static Random random;

    static{
        random = new Random();
    }

    public static double getRandomValue(double fromInclusive, double toInclusive) {
        double diff = toInclusive - fromInclusive;
        return fromInclusive + Math.random() * diff;
    }

    public static double getRandomGaussianValue() {
        return random.nextGaussian();
    }

    public static double calculateSigmoid(double x) {
        if(x >= 10) return 1;
        if(x <= -10) return 0;
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double convertBetween0And1(int value, int min, int max) {
        return ((double)value - min) /(max - min);
    }

    public static <T> void shuffleArray(T[] ar)
    {
        Random rnd = new Random();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            T a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    public static double getCost(double[] desired, double[] actual) {
        assert desired.length == actual.length;
        double sum = 0;
        for(int i = 0; i < desired.length; i++) {
            sum += Math.pow(actual[i] - desired[i], 2);
        }
        return sum;
    }

    public static double getCostDerivative(double actual, double desired) {
        return actual-desired;
    }

    public static double getSigmoidDerivative(double x) {
        var sig = Util.calculateSigmoid(x);
        return sig * (1-sig);
    }

    public static String printDuration(Duration duration) {
        return duration.toString()
                .substring(2)
                .replaceAll("(\\d[HMS])(?!$)", "$1 ")
                .toLowerCase();
    }
}
