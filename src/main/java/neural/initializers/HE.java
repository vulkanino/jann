package neural.initializers;

import java.io.Serializable;

public class HE implements WeightInitializer, Serializable
{
    @Override
    public double[] apply(int numInputs, int numOutputs)
    {
        double stdDev = Math.sqrt(2.0 / numInputs);
        double[] weights = new double[numInputs];
        for (int i = 0; i < numInputs; i++)
            weights[i] = random.nextGaussian() * stdDev;
        return weights;
    }
}

