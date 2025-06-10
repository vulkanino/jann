package neural.initializers;

import java.io.Serializable;

public class Uniform implements WeightInitializer, Serializable
{
    @Override
    public double[] apply(int numInputs, int numOutputs)
    {
        double[] weights = new double[numInputs];
        for (int i = 0; i < numInputs; i++)
            weights[i] = random.nextDouble() - 0.5; // [-0.5, 0.5]
        return weights;
    }
}
