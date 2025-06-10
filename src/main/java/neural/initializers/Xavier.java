package neural.initializers;

import java.io.Serializable;

public class Xavier implements WeightInitializer, Serializable
{
    @Override
    public double[] apply(int numInputs, int numOutputs)
    {
        double limit = Math.sqrt(6.0 / (numInputs + numOutputs));
        double[] weights = new double[numInputs];
        for (int i = 0; i < numInputs; i++)
            weights[i] = (random.nextDouble() * 2 * limit) - limit;
        return weights;
    }
}