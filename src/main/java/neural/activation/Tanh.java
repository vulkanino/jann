package neural.activation;

import java.io.Serializable;

public class Tanh implements ActivationFunction, Serializable
{
    @Override
    public double apply(double input)
    {
        return (Math.exp(input) - Math.exp(-input)) / (Math.exp(input) + Math.exp(-input));
    }

    @Override
    public double derivative(double x)
    {
        double fx = Math.tanh(x);
        return 1.0 - fx * fx;
    }
}
