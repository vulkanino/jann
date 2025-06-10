package neural.initializers;

import java.util.Random;

public interface WeightInitializer
{
    Random random = new Random();
    double[] apply(int numInputs, int numOutputs);
}