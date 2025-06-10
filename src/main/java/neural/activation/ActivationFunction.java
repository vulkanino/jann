package neural.activation;

public interface ActivationFunction
{
    double apply(double input);
    double derivative(double x);
}