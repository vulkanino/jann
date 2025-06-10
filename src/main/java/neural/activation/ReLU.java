package neural.activation;

import java.io.Serializable;

public class ReLU implements ActivationFunction, Serializable
{
    @Override
    public double apply(double input)
    {
        return input <= 0 ? 0 : input;
    }

    @Override
    public double derivative(double x)
    {
        return x > 0.0 ? 1.0 : 0.0;
    }
}
