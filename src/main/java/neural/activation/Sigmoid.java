package neural.activation;

import java.io.Serializable;

public class Sigmoid implements ActivationFunction, Serializable
{
    @Override
    public double apply(double input)
    {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public double derivative(double x)
    {
        double fx = apply(x);
        return fx * (1.0 - fx);
    }
}
