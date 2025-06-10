package main;

import data.DataGrid;
import data.TrainReader;
import neural.NeuralNetwork;
import neural.activation.ActivationFunction;
import neural.activation.ReLU;
import neural.activation.Sigmoid;
import neural.initializers.HE;
import neural.initializers.WeightInitializer;
import neural.initializers.Xavier;
import neural.loss.LossFunction;
import neural.loss.MeanSquaredError;

import java.io.*;
import java.util.*;

/**
 * Uso: java -jar JaNN.jar --mode=output --model=train1.nn --input=mnist_train.csv
 * Uso: java -jar JaNN.jar --mode=input --model=train1.nn
 * Uso: java -jar JaNN.jar --mode=test --model=train1.nn --input=mnist_test.csv
 *
 */
public class Main
{
    private NeuralNetwork neuralNetwork;

    public static void main(String[] args)
    {
        Main m = new Main();
        m.go(args);
    }

    private void go(String[] args)
    {
        Map<String, String> arguments = parseArguments(args);

        String mode = arguments.get("mode");
        if ( mode.equals("output") )
        {
            String modelFile = arguments.get("model");
            addestramento(arguments.get("input"));
            saveNetwork(modelFile);
        }
        else if ( mode.equals("input") )
        {
            String modelFile = arguments.get("model");
            neuralNetwork = loadNetwork(modelFile);
            attesaInput();
        }
        else if ( mode.equals("test") )
        {
            String modelFile = arguments.get("model");
            String testFile = arguments.get("input");
            neuralNetwork = loadNetwork(modelFile);
            runBatchTest(testFile);
        }
        else
            printUsage();
    }

    private Map<String, String> parseArguments(String[] args)
    {
        Map<String, String> map = new HashMap<>();
        for (String arg : args)
        {
            if (arg.startsWith("--") && arg.contains("="))
            {
                String[] parts = arg.substring(2).split("=", 2);
                map.put(parts[0], parts[1]);
            }
        }
        return map;
    }

    private void printUsage()
    {
        System.out.println("Usage for training: java -jar <jar_file> --mode=output --model=train1.nn --input=mnist_train.csv");
        System.out.println("Usage for predicting single digit: java -jar <jar_file> --mode=input --model=train1.nn");
        System.out.println("Usage for predicting batch: java -jar <jar_file> --mode=test --model=train1.nn --input=mnist_test.csv");
        System.exit(1);
    }

    private void addestramento(String trainingFile)
    {
        // lettura del campione di addestramento
        //
        long startTime = System.currentTimeMillis();

        TrainReader tr = new TrainReader(trainingFile);
        System.out.println("Tempo impiegato per la lettura del campione: " + (System.currentTimeMillis() - startTime) + " ms");
        System.out.println("Dimensione campione: " + tr.getDataGrids().size());


        // creazione della rete neurale
        //
        startTime = System.currentTimeMillis();

        final ActivationFunction hiddenAf = new ReLU();
        final ActivationFunction outputAf = new Sigmoid();

        final WeightInitializer hiddenInit = new HE();
        final WeightInitializer outputInit = new Xavier();

        final LossFunction lossFunction = new MeanSquaredError();

        neuralNetwork = new NeuralNetwork(DataGrid.getSize(), lossFunction);
        neuralNetwork.addLayer(128, hiddenInit, hiddenAf);
        neuralNetwork.addLayer(64, hiddenInit, hiddenAf);
        neuralNetwork.addLayer(10, outputInit, outputAf);      // output layer con 10 neuroni
        System.out.println("Tempo impiegato per la creazione della rete neurale: " + (System.currentTimeMillis() - startTime) + " ms");


        // addestramento della rete neurale
        //
        startTime = System.currentTimeMillis();
        neuralNetwork.train(tr.getDataGrids(), 10, 0.01);
        System.out.println("Tempo impiegato per l'addestramento: " + (System.currentTimeMillis() - startTime) + " ms");
    }

    private void attesaInput()
    {
        Scanner scanner = new Scanner(System.in);

        while ( true )
        {
            System.out.println("Inserisci 784 valori separati da spazio, virgola o a capo:");
            List<Double> inputList = new ArrayList<>(784);

            while (inputList.size() < 784)
            {
                String line = scanner.nextLine().trim();
                if ( line.isEmpty() )
                    continue;

                String[] tokens = line.split("[,\\s]+"); // spazio, virgola, tab o newline
                for (String token : tokens)
                {
                    if (!token.isEmpty())
                    {
                        try
                        {
                            inputList.add(Double.parseDouble(token) / 255.0);
                        }
                        catch (NumberFormatException e)
                        {
                            System.out.println("Valore non valido: " + token);
                        }
                    }
                }
            }

            double[] input = inputList.stream().mapToDouble(Double::doubleValue).toArray();
            double[] output = neuralNetwork.predict(input);

            /* Elegante ma meno efficiente
            int predicted = IntStream.range(0, output.length)
                    .boxed()
                    .max(comparingDouble(i -> output[i]))
                    .orElse(-1);
             */

            int predicted = 0;
            for (int i = 1; i < output.length; i++)
            {
                if (output[i] > output[predicted])
                    predicted = i;
            }
            System.out.println("La rete neurale ha predetto: " + predicted);
        }
    }

    void saveNetwork(String filename)
    {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename)))
        {
            out.writeObject(neuralNetwork);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    NeuralNetwork loadNetwork(String filename)
    {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename)))
        {
            return (NeuralNetwork) in.readObject();
        } catch (IOException | ClassNotFoundException e)
        {
            throw new RuntimeException(e);
        }
    }

    void runBatchTest(String filename)
    {
        TrainReader testReader = new TrainReader(filename);
        List<DataGrid> testSet = testReader.getDataGrids();

        int correct = 0;
        long totalTime = 0;

        System.out.printf("%-5s %-7s %-10s %-10s %-10s%n", "#", "Label", "Predicted", "Correct?", "Time(ms)");

        for (int i = 0; i < testSet.size(); i++)
        {
            DataGrid sample = testSet.get(i);
            double[] input = sample.getGrid();
            int expected = sample.getLabel();

            long start = System.nanoTime();
            double[] output = neuralNetwork.predict(input);
            long end = System.nanoTime();

            int predicted = argMax(output);
            boolean match = predicted == expected;
            long timeMs = (end - start) / 1_000_000;

            if ( match )
                correct++;
            totalTime += timeMs;

            System.out.printf("%-5d %-7d %-10d %-10s %-10d%n", i, expected, predicted, match ? "YES" : "NO", timeMs);
        }

        double accuracy = (100.0 * correct) / testSet.size();
        double avgTime = (double) totalTime / testSet.size();

        System.out.printf("Accuracy: %.2f%% (%d/%d)%n", accuracy, correct, testSet.size());
        System.out.printf("Tempo medio: %.2f ms per predizione%n", avgTime);
    }

    private int argMax(double[] array)
    {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++)
        {
            if (array[i] > max)
            {
                max = array[i];
                index = i;
            }
        }
        return index;
    }
}
