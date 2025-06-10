![image](https://github.com/user-attachments/assets/d9b7dffd-6873-4b02-b50a-6ac6848aa31f)![image](https://github.com/user-attachments/assets/86602de8-38f4-435f-97a8-0d6c6e273e70)<img width="482" alt="image" src="https://github.com/user-attachments/assets/9e9a13b3-8e08-45d4-98f6-a6148210d3c7" /># JaNN
Una rete neurale completamente connessa codificata in Java per il riconoscimento di cifre scritte a mano. 

## Introduzione ##

Le reti neurali artificiali sono modelli matematici ispirati al funzionamento del cervello umano, progettati per riconoscere schemi e prendere decisioni in base ai dati. Una delle architetture più semplici ma potenti è la rete neurale completamente connessa (fully connected neural network), nella quale ogni neurone di un livello è connesso a tutti i neuroni del livello successivo.

Con questo articolo implementeremo gradualmente una semplice rete neurale completamente connessa usando Java moderno (versione 24), atta al riconoscimento di numeri scritti a mano. 

## Apprendimento
Per l’apprendimento della rete useremo un file da 60.000 campioni. Sia il file campione sia quello di test possono essere scaricati da qui: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download 

 <img width="482" alt="image" src="https://github.com/user-attachments/assets/a65211d8-37e0-43a0-90dd-5bee2bcce3ea" />

Il file campione è un CVS costituito da righe; ciascuna riga contiene come primo valore una label, cioè la cifra nota e rappresentata dai valori successivi. I valori successivi sono un vettore con la rappresentazione orizzontale di una griglia da 28x28 pixel in cui ci sono valori da 0 a 255, dove lo 0 corrisponde al pixel nero, il 255 corrisponde al pixel bianco, e tutti i valori intermedi di grigio. 

Ad esempio un possibile “5” scritto a mano nella griglia può essere così:

<img width="346" alt="image" src="https://github.com/user-attachments/assets/70846f60-587b-461c-9ab7-4f413d1c6b69" />

Mentre un altro “5” scritto un po’ male può essere così:

<img width="338" alt="image" src="https://github.com/user-attachments/assets/871077f8-8fe0-4a43-9310-f07d1095c63c" />

Una rete completamente connessa può apprendere a classificare correttamente una cifra da 0 a 9 scritta a mano dentro una griglia 28x28. Si tratta di una classificazione multiclasse, visto che abbiamo 10 cifre da riconoscere. 

Ma prima di scrivere del codice, ci tocca un po’ di teoria. 

## Neurone
<img width="243" alt="image" src="https://github.com/user-attachments/assets/1c843be4-30a6-407a-b2d5-a918b970cee6" />

Si può pensare al neurone come ad un'unità computazionale. Riceve in input un vettore di valori (nel nostro esempio riceverà inizialmente i 28x28=784 pixel dell’immagine che rappresenta la cifra scritta a mano). Quindi moltiplica quei valori per dei pesi e somma il risultato a un bias, ottenendo così un valore grezzo chiamato logit. Il valore viene poi passato ad una funzione di attivazione (non lineare, importante!) per ottenere il singolo output finale. La formula che descrive il comportamento del neurone si può scrivere così:

![image](https://github.com/user-attachments/assets/aaef6273-8883-4933-8284-a9e32942b2df)

Cioè: passiamo alla funzione di attivazione ϕ la sommatoria di tutti i valori X moltiplicati per i propri pesi W, più il bias B. Si può scrivere la formula più praticamente anche in forma vettoriale, ma il senso non cambia. 

## A che servono i pesi?
Durante l'apprendimento, i pesi (e i bias) sono i parametri che la rete deve modificare per ridurre l'errore (loss) tra l'output prodotto e quello atteso. Se il risultato del neurone è diverso da quello desiderato, la rete calcola quanto ogni peso ha contribuito all’errore (attraverso la retropropagazione del gradiente) e li aggiorna di conseguenza. Questo processo è guidato da una funzione di perdita, che quantifica la distanza tra predizione e realtà. In sostanza, la rete deve premiare i risultati positivi e scoraggiare quelli negativi. Quindi, i pesi e i bias sono le uniche variabili che la rete può e deve modificare per imparare.

## A che serve il bias, non bastavano i pesi? 
No, i pesi da soli non bastano. Il bias è un termine aggiuntivo che permette al neurone di traslare la funzione di attivazione lungo l’asse orizzontale. In altre parole, consente alla rete di spostare la soglia di attivazione indipendentemente dagli input. Immagina il neurone come un interruttore della luce che si accende solo se riceve abbastanza corrente (input). Senza il bias, la soglia di accensione è sempre zero. Con il bias, possiamo regolare quella soglia: possiamo far accendere la luce solo dopo un certo livello di input, o anche subito con input minimi. Matematicamente, senza bias, la funzione del neurone è vincolata a passare per l’origine del grafico (cioè il punto 0,0). Questo limita fortemente la capacità della rete di modellare funzioni complesse. Il bias agisce come una costante di traslazione, migliorando la flessibilità della rete nel trovare una soluzione che minimizzi l’errore.

## Perché serve una funzione di attivazione? 
Se usassimo solo somme pesate e bias, cioè operazioni lineari, anche l’intera rete (composta da più neuroni e layer) sarebbe una funzione lineare. Ma delle funzioni lineari non bastano per risolvere problemi complessi come riconoscere cifre scritte a mano o classificare immagini. La funzione di attivazione introduce non linearità: in questo modo, la rete può modellare relazioni complesse tra input e output. Serve a trasformare il logit (la somma pesata più bias) in un output attivo o filtrato, che rappresenta una decisione o una risposta intermedia.

## Perché dev’essere non lineare?
Perché la composizione di funzioni lineari è ancora una funzione lineare. Se non usiamo non-linearità, anche una rete con dieci layer farà lo stesso lavoro di un solo layer: una semplice somma pesata. L’intera rete sarebbe una semplice regressione lineare, inutile per problemi reali. 

Esempi di funzioni non lineari:


1  La ReLU è una funzione semplicissima ma molto comune (nei layer nascosti, spiegati più avanti) proprio perché è semplice e accelera l'apprendimento. Se il numero ricevuto è minore o uguale a zero, lo forza a zero; altrimenti ritorna il numero stesso.

f(x)=max⁡(0,x)

2  La softmax, usata specialmente in output per classificazione multiclasse, è una funzione vettoriale che restituisce una distribuzione di probabilità. Valori più grandi diventano esponenzialmente più probabili. 
![image](https://github.com/user-attachments/assets/7ace65f0-6c6d-4c63-ada2-35aa738775e2)


3  La sigmoid trasforma un numero reale in un valore compreso tra 0 e 1. La sua forma è a “S” morbida e crescente. Spesso usata nei neuroni di output per la classificazione binaria. 

σ(x)=1/(1+e^(-x) )


4  La tanh (iperbolica tangente), simmetrica rispetto all’origine (a differenza della sigmoid), è buona per dati normalizzati intorno a zero (ed è usata spesso nei layer nascosti), specialmente prima della diffusione della ReLU.

tanh⁡(x)=  (e^x- e^(-x))/(e^x+e^(-x) )

## Layer
<img width="356" alt="image" src="https://github.com/user-attachments/assets/273bc5f5-07be-40b4-8719-dbb09fc15430" />

Un layer è un insieme di neuroni che si può pensare disposto in verticale. La rete neurale è normalmente costituita da più layer, contenenti ciascuno un numero decrescente di neuroni fino ad arrivare all’ultimo chiamato output layer. Layer intermedi tra input e output si chiamano layer nascosti (hidden layer). Ogni layer trasforma il proprio input in una rappresentazione più astratta, utile alla classificazione. Il layer di output fornisce la predizione finale. 

## Forward Pass
Il forward pass (o propagazione in avanti) è il processo attraverso cui i dati scorrono dalla rete:

1. I pixel normalizzati (tra 0 e 1) entrano nel layer di input.
2. Ogni layer attraverso i neuroni calcola tutti i logit.
3. I logit, sempre attraverso i neuroni, passano attraverso una funzione di attivazione, come la ReLU nei layer nascosti o la softmax nel layer di output.
4. L’output finale è un vettore di probabilità per ciascuna cifra (0–9).

## Loss Function
La loss function (funzione di perdita) misura quanto l'output della rete differisce dalla label corretta. Per la classificazione si usa spesso la cross-entropy loss, che penalizza le predizioni lontane dalla vera classe.

## Il gradiente e la sua regola 
Il gradiente della perdita è come una bussola: ti dice quanto e in quale direzione cambiare ogni peso per diminuire l’errore. La regola del gradiente discendente è un algoritmo di ottimizzazione che serve a modificare i pesi (e i bias) dei neuroni in una rete neurale per ridurre l’errore tra l’output previsto e quello desiderato.

Immagina che l'errore della rete (la loss) sia un paesaggio collinare. Tu vuoi scendere verso il punto più basso, cioè dove l'errore è minimo. Per farlo, devi capire in che direzione andare. Quella direzione è data dal gradiente. I passi da seguire sono:

1  Calcola il gradiente della funzione di errore rispetto a ciascun peso (quanto cambia l’errore se cambi quel peso).
2  Aggiorna ogni peso spostandolo un po’ nella direzione opposta al gradiente (da qui “discendente”).
3  La quantità di spostamento è regolata da un parametro chiamato learning rate (quanto “grande” è ogni passo).

In pratica, se un peso contribuisce troppo all’errore, lo riduci un po’. Se contribuisce troppo poco lo aumenti un po’. Così, passo dopo passo, la rete impara!

## Backward pass ed epoche
Il backward pass (o backpropagation) calcola quanto ogni peso ha contribuito all'errore:

1  Si calcola il gradiente del loss rispetto ai pesi usando la derivata della funzione di attivazione.
2  Si aggiornano i pesi nella direzione opposta al gradiente (gradiente discendente), riducendo l'errore.

Il ciclo di forward + loss + backpropagation + aggiornamento pesi costituisce un'epoca di training. Dopo molte epoche, la rete apprende a classificare correttamente.

## Caricamento dati di apprendimento
Passiamo ad una codifica che permetta di caricare i dati di apprendimento in una struttura di dati idonea. Scriviamo per prima cosa la classe che memorizza una singola riga del file CSV di apprendimento, cioè la label e la griglia 28x28 di valori di grigio.

```
public class DataGrid
{
    private final int label;
    private final double[] grid;

    private static final int ROWS=28;
    private static final int COLS=28;

    public DataGrid(String dataLine)
    {
        String[] data = dataLine.split(",");
        assert data.length == ROWS * COLS + 1;

        this.label = Integer.parseInt(data[0]);
        this.grid = new double[ROWS * COLS];
        for (int i = 1; i < ROWS * COLS; i++)
            grid[i] = Integer.parseInt(data[i]) / 255.0;
    }

    public int getLabel()
    {
        return label;
    }

    public double[] getGrid()
    {
        return grid;
    }

    public static int getSize()
    {
        return ROWS * COLS;
    }
}
```

Anche se i valori da leggere sono dei semplici byte, preferisco memorizzarli in un array di double perché la rete neurale lavora meglio su numeri con più alta precisione, per cui anziché normalizzarli in un secondo momento, perdendo efficienza, li carichiamo direttamente come double e non ci pensiamo più. 

### SPAZIO NERD
Eccoci di fronte al primo compromesso: efficienza o sicurezza? La classe così com’è non è immutabile: getGrid() infatti ritorna l’array così com’è e quindi può essere modificato da un chiamante. La soluzione sicura sarebbe:

```
public double[] getGridCopy()
{
    return Arrays.copyOf(grid, grid.length);
}
```

Ma ad ogni chiamata verrebbe allocato un nuovo array double[784] con un costo stimato di 1 o 2 microsecondi per chiamata, più l’impegno della garbage collection se usata spesso. È pochissimo ma su grandi campioni ogni microsecondo si fa sentire. La tengo così, preferisco documentare nel codice di non modificare l’array. Sarebbe perfetto se fosse Java a permettere di dichiarare immutabile il ritorno di un metodo, magari con final, ma in Java questa parola chiave applicata sul metodo ha ben altro significato; diciamo che i progettisti all’inizio erano più focalizzati sull’ereditarietà che sulla robustezza. Altri linguaggi che lo permettono: Kotlin, Rust, C++, Python, perfino JavaScript!

Ora ci serve una classe che legge tutte le righe e le inserisce in una struttura di dati, useremo internamente un ArrayList.

```
public class TrainReader
{
    private final List<DataGrid> dataGrids;

    public TrainReader(String filePath)
    {
        // Primo passaggio: conta il numero di righe
        //
        final int lineCount = calcolaNumeroRighe(filePath);

        // Secondo passaggio: crea una lista di DataGrid
        //
        dataGrids = new ArrayList<>(lineCount);  
        try (var reader = new java.io.BufferedReader(new FileReader(filePath)))
        {
            reader.lines()
                    .skip(1) // Salta l'intestazione
                    .map(DataGrid::new)
                    .forEach(dataGrids::add);
        }
        catch (IOException e)
        {
            System.err.println("Errore nella lettura del file: " + e.getMessage());
            exit(1);
        }
    }

    private int calcolaNumeroRighe(String filePath)
    {
        try (var reader = new BufferedReader(new FileReader(filePath)))
        {
            return (int) reader.lines().count() - 1; // Escludi l'intestazione
        }
        catch (IOException e)
        {
            System.err.println("Errore nella lettura del file: " + e.getMessage());
            exit(1);
            return 0;
        }
    }

    public List<DataGrid> getDataGrids()
    {
        return dataGrids;
    }
}
```

Per leggere il file campione possiamo ora scrivere:

```
TrainReader tr = new TrainReader("mnist_train.csv");
```

## Codifica del neurone
Ci serve una classe che implementi il neurone. Iniziamo con i soli pesi e il bias, così:

```
class Neuron
{
    private final double[] weights;
    private double bias;

    public Neuron(int inputSize)
    {
        this.weights = new double[inputSize];
        this.bias = 0.0;
    }
}
```

Ora scriviamo un metodo che inizializza i pesi e il bias con valori casuali. 

```
private void initializeWeights()
{
    // Inizializza bias casuale tra -0.5 e 0.5
    bias = random.nextDouble() - 0.5; 
    for (int i = 0; i < weights.length; i++)
        // Inizializza pesi casuali tra -0.5 e 0.5
        weights[i] = random.nextDouble() - 0.5; 
}
```

Qui c’è un piccolo problema tecnico: è cruciale che questi valori siano distribuiti in maniera uniforme e specializzata, e la nextDouble non è ottimale per reti più profonde o per classificazioni più complesse. Dobbiamo generare numeri in modo migliore, perché se i pesi sono troppo piccoli, i segnali che viaggiano nella rete tendono a diventare anch’essi troppo piccoli (vanishing gradients). Se invece sono troppo grandi, rischiano di saturare le funzioni di attivazione (specie sigmoid o tanh), bloccando l'apprendimento.

Migliori strategie possono essere la Xavier/Grolot, che è perfetta per le funzioni di attivazione come sigmoid o tanh, oppure la He inizialization (distribuzione normale gaussiana centrata in 0 con deviazione standard adattiva), perfetta per la ReLU.

Dal momento che la funzione di distribuzione ideale dipende anche dalla funzione di attivazione usata, mi piacerebbe se tale funzione di distribuzione potesse essere inviata al neurone dal chiamante, cosa che permetterebbe anche di cambiarla a caldo in base alle necessità. Riesci qui a identificare il design pattern che permette di modificare un comportamento a caldo? 

Scriviamo l’interfaccia per gli inizializzatori di pesi:

```
public interface WeightInitializer
{
    Random random = new Random();

    double[] apply(int numInputs, int numOutputs);
}
```

E ora implementiamo tre inizializzatori:

```
public class Uniform implements WeightInitializer
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

public class Xavier implements WeightInitializer
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

public class HE implements WeightInitializer
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
```

Si può notare come il primo, Uniform, sia identico a quello che avevamo scritto inizialmente. Aggiorniamo ora la classe Neuron affinché accetti un inizializzatore di pesi nel costruttore:

```
public class Neuron
{
    private static Random random = new Random();

    private double bias;
    private final double[] weights;

    public Neuron(int inputSize, int outputSize, WeightInitializer initializer)
    {
        this.bias = random.nextDouble() - 0.5; // Inizializza bias casuale tra -0.5 e 0.5.
        this.weights = initializer.apply(inputSize, outputSize);
    }
}
```

Non abbiamo ancora finito col neurone. Anche la funzione di attivazione vorrei potesse essere passata al neurone dal chiamante ed eventualmente cambiata a caldo, a seconda del layer dove si troverà il neurone. Infatti, alcune funzioni di attivazione sono più indicate per gli input layer, altre per gli hidden e altre ancora per l’output  layer. Perciò iniziamo con lo scrivere l’interfaccia per le funzioni di attivazione:

```
public interface ActivationFunction
{
    double apply(double input);
}
```

E qualche semplice implementazione (vedi note teoriche sopra):

```
public class ReLU implements ActivationFunction
{
    @Override
    public double apply(double input)
    {
        return input <= 0 ? 0 : input;
    }
}

public class Sigmoid implements ActivationFunction
{
    @Override
    public double apply(double input)
    {
        return 1.0 / (1.0 + Math.exp(-input));
    }
}

public class Tanh implements ActivationFunction
{
    @Override
    public double apply(double input)
    {
        return (Math.exp(input) - Math.exp(-input)) / (Math.exp(input) + Math.exp(-input));
    }
}
```

Bene, aggiorniamo Neuron modificando il costruttore, aggiungendo anche il metodo che calcola l’output, quello per aggiornare i pesi durante l’elaborazione e anche il metodo per impostare la funzione di attivazione a caldo, le parti interessanti in grassetto:

```
public class Neuron
{
    private static Random random = new Random();

    private double bias;
    private final double[] weights;
    private ActivationFunction activationFunction;

    public Neuron(int inputSize, int outputSize, WeightInitializer initializer, ActivationFunction activationFunction)
    {
        this.bias = random.nextDouble() - 0.5; // Inizializza bias casuale tra -0.5 e 0.5.
        this.weights = initializer.apply(inputSize, outputSize);
        this.activationFunction = activationFunction;
    }

    public double computeOutput(double[] inputs)
    {
        checkInputs(inputs);

        double sum = bias;
        for (int i = 0; i < inputs.length; i++)
            sum += inputs[i] * weights[i];

        return activationFunction.apply(sum);
    }

    public void updateWeights(double[] inputs, double gradient, double learningRate)
    {
        checkInputs(inputs);

        // Aggiorna i pesi
        for (int i = 0; i < weights.length; i++)
            weights[i] -= learningRate * gradient * inputs[i];

        // Aggiorna il bias
        bias -= learningRate * gradient;
    }

    private void checkInputs(double[] inputs)
    {
        if ( inputs.length != weights.length )
            throw new IllegalArgumentException("La dimensione degli input non corrisponde a quella dei pesi");
    }

    public void setActivationFunction(ActivationFunction activationFunction)
    {
        this.activationFunction = activationFunction;
    }
}
```

## Codifica del Layer
Abbiamo detto che un layer è costituito da un certo numero di neuroni; quanti saranno questi layer e di quanti neuroni saranno costituiti sarà il chiamante a deciderlo. Il layer deve solo calcolare gli output di ogni neurone. Il suo chiamante si occuperà di passare questi output al layer successivo. 

```
class Layer
{
    private final List<Neuron> neurons;

    public Layer(int inputSize, int neurons, WeightInitializer initializer, ActivationFunction activationFunction)
    {
        this.neurons = new ArrayList<>(neurons);
        for (int i = 0; i < neurons; i++)
            this.neurons.add(new Neuron(inputSize, neurons, initializer, activationFunction));
    }

    /**
     * Calcola l'output di questo layer dato l'input.
     */
    public double[] computeOutput(double[] input)
    {
        return neurons
            .stream()
            .mapToDouble(neuron -> neuron.computeOutput(input))
            .toArray();

    }

    public List<Neuron> getNeurons()
    {
        return neurons;
    }
}
```

Nel costruttore si creano i neuroni passando l’inizializzatore dei pesi e la funzione di attivazione. 

### SPAZIO NERD
Il metodo che calcola gli output di tutti i neuroni usa gli stream, però si può migliorare al fine di ottenere migliori prestazioni. Il problema non sta tanto del doppio ciclo, come si potrebbe credere (chiamata a stream più toArray) perché tutto il blocco viene eseguito in un unico passaggio, grazie alla lazy evaluation degli stream in Java. C’è solo un piccolo sovraccarico dovuto all’uso stesso dello stream, quindi preferisco usare un normale ciclo in questo caso. 

A cosa è dovuto il sovraccarico? Ogni .stream() crea un oggetto Stream, che ha uno stato interno e uno scheduler di operazioni. Anche mapToDouble() crea un'ulteriore pipeline di trasformazione.  Ogni chiamata passa attraverso una macchina a stati. Ogni neuron.computeOutput è “wrappata” in una lambda che viene inserita in una pipeline (oggetto intermedio), che richiede una serie di metodi virtuali e controlli interni. Anche toArray() ha un sovraccarico extra rispetto al semplice array double[] preallocato nel for. Insomma, nel ciclo for le istruzioni sono più prevedibili e lineari in memoria. 

Sarà meno elegante, ma cambiamo il metodo di calcolo:

```
public double[] computeOutput(double[] input)
{
    double[] output = new double[neurons.size()];
    for (int i = 0; i < neurons.size(); i++)
        output[i] = neurons.get(i).computeOutput(input);

    return output;
}
```

## Codifica della rete neurale
Ora ci serve un gestore a cappello che crea i layer e li gestisce. Iniziamo con una bozza.

```
class NeuralNetwork
{
    private final List<Layer> layers;
    private final int inputSize;

    public NeuralNetwork(int inputSize)
    {
        this.layers = new ArrayList<>();
        this.inputSize = inputSize;
    }

    public void addLayer(int numberOfNeurons, WeightInitializer initializer, ActivationFunction activationFunction)
    {
        // determina la dimensione dell'input del nuovo layer in base al numero di neuroni dell'ultimo layer
        final int currentInputSize = (layers.isEmpty()) ? inputSize : layers.get(layers.size() - 1).getNeurons().size();
        final Layer layer = new Layer(currentInputSize, numberOfNeurons, initializer, activationFunction);
        layers.add(layer);
    }
}
```

Il costruttore crea un elenco di layer vuoto. Quando il chiamante eseguirà addLayer, questo metodo creerà un nuovo layer calcolando la dimensione degli input. 

Ora ci serve un metodo fondamentale in grado di predire la cifra scritta a mano, cioè che prenda gli output di ogni layer e lo passi come input al layer successivo, fino alla fine.

```
public double[] predict(double[] input)
{
    return layers
            .stream()
            .reduce(input,
                    (currentInput, layer) -> layer.computeOutput(currentInput),
                    (a, b) -> b); // questo combinatore non serve ma è richiesto dalla reduce
}
```

L’uso della reduce come funzione terminale qui è perfetto, perché passa l'output di un layer come input del successivo, fino ad arrivare all’output finale della rete. Nel nostro caso input è il primo array in ingresso alla rete. Alla fine viene restituito l’output dell’ultimo layer. In un mondo non funzionale corrisponderebbe a (quante volte l’avete visto?):

```
double[] result = input;
result = L1.computeOutput(result); // output1
result = L2.computeOutput(result); // output2
result = L3.computeOutput(result); // output3
return result;
```

Questo metodo è uno dei pilastri della rete neurale, consente di passare la palla da layer a layer fino ad arrivare in fondo. Il metodo per il momento non tiene conto della propagazione all’indietro, ma ci arriveremo presto. 

## Calcolo della perdita 
Al fine di addestrare la rete, ci serve un metodo che calcoli l’errore (loss) in modo che si possano correggere i pesi di conseguenza. In maniera simile a quanto fatto per le funzioni di attivazione, anche in questo caso voglio che la funzione di calcolo dell’errore sia parametrico e sostituibile a caldo, quindi definisco anzitutto un’interfaccia:

```
public interface LossFunction
{
    // calcola il valore del loss
    double compute(double[] predicted, double[] expected);
}
```

Cioè questa interfaccia vuole che il metodo di calcolo accetti un vettore di valori così come predetti dalla rete e uno con i valori attesi. Il ritorno sarà un numero che indica la distanza tra questi due insiemi.

Ora serve almeno un’implementazione; proviamo a scrivere l’algoritmo Mean Squared Error. Cos’è? La MSE è una funzione di loss che misura quanto è lontano l’output della rete da quello che dovrebbe essere (il “target”). La formula è:

![image](https://github.com/user-attachments/assets/050ac31f-c99b-41c5-ab1e-9ed80408d3b3)

Dove:
	y_(pred,i) è l’i-esimo valore previsto dal neurale (es. 0,07).
	y_(true,i) è l’i-esimo valore atteso (es. 1,0).
	Si somma il quadrato della differenza per tutti i valori, poi si fa la media.

Con le formule sembra tutto più complicato di quello che è, mentre è veramente semplice: fa la media dei quadrati delle differenze. La MSE si chiama “squared” perché usa il quadrato dell’errore; serve per penalizzare più fortemente gli errori grandi ed evita che errori positivi e negativi si annullino tra loro.

L’implementazione è un gioco da ragazzi:

```
public class MeanSquaredError implements LossFunction
{
    @Override
    public double compute(double[] predicted, double[] expected)
    {
        checkInputs(predicted, expected);

        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++)
        {
            double diff = predicted[i] - expected[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }

    private void checkInputs(double[] predicted, double[] expected)
    {
        if ( predicted.length != expected.length )
            throw new IllegalArgumentException("Dimensioni diverse tra predicted e expected");
    }
}
```

Manca un pezzo però. Quando implementeremo la propagazione all’indietro avremo bisogno anche della derivata della funzione di loss. Tantovale metterla da subito dentro l’interfaccia. 

```
public interface LossFunction
{
    // calcola il valore del loss
    double compute(double[] predicted, double[] expected);

    // restituisce il gradiente del loss rispetto all’output della rete (necessario per la backpropagation)
    double[] derivative(double[] predicted, double[] expected);
}
```

Quindi implementiamola per MSE, così siamo pronti per la propagazione all’indietro. Infatti per aggiornare i pesi servono i gradienti. La derivata della MSE rispetto all’output previsto è:

![image](https://github.com/user-attachments/assets/0e217d43-c50f-4403-8c94-a928fdda2fa0)

Serve per capire quanto e in che direzione correggere l’output della rete per avvicinarlo al target. Ecco la semplice implementazione in Java:

```
public class MeanSquaredError implements LossFunction
{
    // … omesso compute e checkInputs

    @Override
    public double[] derivative(double[] predicted, double[] expected)
    {
        checkInputs(predicted, expected);

        double[] grad = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++)
            grad[i] = 2 * (predicted[i] - expected[i]) / predicted.length;

        return grad;
    }
}
```

Ora possiamo aggiornare la rete perché durante la fase di addestramento usi una funzione loss:

```
class NeuralNetwork
{
    private final List<Layer> layers;
    private final int inputSize;
    private final LossFunction lossFunction;


    public NeuralNetwork(int inputSize, LossFunction lossFunction)
    {
        this.layers = new ArrayList<>();
        this.inputSize = inputSize;
        this.lossFunction = lossFunction;
    }

    // omessa addLayer e predict per brevità

    public void train(double[] input, double[] expected, double learningRate)
    {
        double[] output = predict(input);

        // Calcola il loss  
        double loss = lossFunction.compute(output, expected);
        System.out.println("Loss: " + loss);

        // Calcola il gradiente dell’output layer
        double[] outputGradient = lossFunction.derivative(output, expected);

        // Qui manca la retropropagazione
     }
}
```

Il metodo train è incompleto perché non implementa la retropropagazione, ma ci arriviamo.

## Rivediamo ActivationFunction
Avevamo creato l’interfaccia ActivationFunction con il solo metodo apply. Come fatto per la LossFunction è opportuno dotare l’interfaccia anche del metodo che calcola la derivata, così leghiamo ad ogni funzione di attivazione la sua specifica derivata. Il neurone che ha già ricevuto una funzione di attivazione sarà automaticamente in grado di chiamare anche la sua derivata per poter aggiornare i propri pesi. 

.  La derivata di ReLU è: ritorna 1 se x > 0, altrimenti ritorna 0
.  La derivata di Sigmoid è: f(x)*(1-f(x))
.  La derivata di tanh è: f^' (x)=1-  〖tan^2 (〗⁡x  )

È più semplice da implementare che da descrivere: 

```
public interface ActivationFunction
{
    double apply(double input);
    double derivative(double x);
}

public class ReLU implements ActivationFunction
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

public class Sigmoid implements ActivationFunction
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

public class Tanh implements ActivationFunction
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
```

## Retropropagazione 
Abbiamo visto che, durante la fase di addestramento, per ogni input otteniamo l’output grazie alla propagazione in avanti. Siamo già in grado di calcolare l’errore usando una funzione di perdita LossFunction. Adesso dobbiamo:

1. Calcolare il gradiente del loss rispetto all’output.
2. Propagare quel gradiente a ritroso.
3. Aggiornare i pesi di ogni neurone con la regola del gradiente discendente (vedi paragrafo teorico).

Per far questo ci servono due dati in più nel neurone: 1) l’ultimo input ricevuto e 2) la somma pesata prima dell’applicazione della funzione di attivazione. Dobbiamo salvare internamente questi due dati, in questo modo saremo in grado di aggiornare i pesi usando il gradiente e il tasso di apprendimento, il learning rate. 

Chiamiamo lastZ la somma pesata prima dell’attivazione (già calcolata, sarebbe il logit): 

![image](https://github.com/user-attachments/assets/d6a73a96-c220-4723-ac4b-bd2abcd850f8)

Per calcolare l’output del neurone abbiamo fatto:

output=f(z)

Perché serve lastZ nella retropropagazione? Perché per calcolare il gradiente della funzione di attivazione serve la derivata rispetto a z, cioè f'(z). Non ci serve il valore attivato, cioè l'output del neurone, ma solo z, perché le derivate delle funzioni di attivazione si esprimono in funzione di z, non direttamente in funzione degli input grezzi.

Con ciò possiamo aggiornare il neurone:

```
public class Neuron
{
    // … qui come prima

    // Per la backpropagation
    private double[] lastInput;
    private double lastZ;


    public double computeOutput(double[] inputs)
    {
        checkInputs(inputs);

        this.lastInput = inputs;
        this.lastZ = bias;
        for (int i = 0; i < inputs.length; i++)
            lastZ += inputs[i] * weights[i];

        return activationFunction.apply(lastZ);
    }

    public void updateWeights(double gradient, double learningRate)
    {
        for (int i = 0; i < weights.length; i++)
            weights[i] -= learningRate * gradient * lastInput[i];

        bias -= learningRate * gradient;
    }

    // altri metodi omessi
}
```

Ora possiamo aggiornare anche Layer per abilitare la propagazione all’indietro:

```
public double[] backward(double[] dLoss_dA, double[] prevActivations, double learningRate)
{
    int numInputs = prevActivations.length;
    int numNeurons = neurons.size();
    double[] dLoss_dA_prev = new double[numInputs]; // da restituire

    for (int j = 0; j < numNeurons; j++)
    {
        Neuron neuron = neurons.get(j);

        double z = neuron.getLastZ(); // z_j = somma pesata prima di attivazione
        double da_dz = neuron.getActivationFunction().derivative(z); // ∂A/∂Z

        double dLoss_dZ = dLoss_dA[j] * da_dz; // ∂L/∂Z = ∂L/∂A * ∂A/∂Z

        neuron.updateWeights(dLoss_dZ, learningRate);

        // Calcola contributo all’errore del layer precedente: dL/dA_prev = somma( w_jk * dL/dZ_j )
        double[] weights = neuron.getWeights();
        for (int k = 0; k < weights.length; k++)
        {
            dLoss_dA_prev[k] += weights[k] * dLoss_dZ;
        }
    }

    return dLoss_dA_prev;
}
```

Per capire cosa fa esattamente bisogna fare delle premesse. Abbiamo già detto che ogni layer è come una fila di neuroni e che ogni neurone ha dei pesi (come manopole che regolano l’intensità di ogni input) e una funzione di attivazione (che decide quanto “attivo” sarà il neurone). 

Il metodo backward ha due scopi:

1. Correggere i pesi dei neuroni (cioè imparare dagli errori).
2. Calcolare quanto errore deve essere “passato indietro” al layer precedente, per continuare la correzione.

Riceve in input questi dati:

. dLoss_dA: l'errore sul risultato di questo layer (∂L/∂A)  
. prevActivations: output del layer precedente che ci serve per aggiornare i pesi
. learningRate: iperparametro di aggiornamento, cioè quanto velocemente imparare

E per ogni neurone del layer:
1. Prende z (cioè il valore prima della funzione di attivazione, salvato durante il forward pass).
2. Calcola quanto la funzione di attivazione cambia in quel punto (cioè la derivata).
3. Usa questo valore per trasformare l’errore rispetto all’output (∂L/∂A) in un errore rispetto a z (∂L/∂Z).
4. Usa questo ∂L/∂Z per aggiornare i pesi del neurone (cioè correggere l’apprendimento).
5. Usa i pesi per calcolare quanto errore ogni input ha causato, così da passarlo al layer precedente.
 
Infine ritorna l’errore da passare al layer precedente. Questo è cruciale: quando questo layer ha finito di correggersi, dice al layer prima di lui: “Questa era la mia parte di colpa, ora tocca a voi sistemare i vostri pesi.”

Questo metodo è, insieme a predict, il cuore della rete.

## Addestramento
Rivediamo la classe NeuralNetwork per modificare il metodo train con la retropropagazione. Alla fine di un ciclo di addestramento, la rete produce un output come ad esempio: 

[0.01, 0.02, 0.05, 0.89, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]

Questo indica che secondo la rete la cifra è molto probabilmente un 3. Per calcolare il loss, dobbiamo confrontare questo output con il vettore atteso che supponiamo sia proprio 3. Dobbiamo perciò mettere sotto la stessa forma vettoriale questo valore atteso, cioè qualcosa come: 

[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

È simile al concetto delle subnet mask per gli indirizzi IP. Questo vettore si chiama “One hot vector”, ha attivo solo il neurone con il valore atteso. Scriviamo il semplice metodo che costruisce questo vettore a partire dalla label.

```
private double[] toOneHot(int label)
{
    double[] oneHot = new double[10];
    oneHot[label] = 1.0;
    return oneHot;
}
```

Possiamo ora completare il metodo train aggiungendo la retropropagazione:

```
public void train(List<DataGrid> trainingSet, int epochs, double learningRate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalLoss = 0.0;

        for (DataGrid sample : trainingSet)
        {
            double[] input = sample.getGrid();
            double[] target = toOneHot(sample.getLabel());

            // Forward pass
            List<double[]> activations = new ArrayList<>(layers.size() + 1);
            activations.add(input);
            double[] current = input;
            for (Layer layer : layers)
            {
                current = layer.computeOutput(current);  // salva lastInput e lastZ nel layer
                activations.add(current);
            }

            // Loss
            totalLoss += lossFunction.compute(current, target);

            // Backward pass
            double[] delta = lossFunction.derivative(current, target); // iniziale dL/dA

            for (int i = layers.size() - 1; i >= 0; i--)
            {
                Layer layer = layers.get(i);
                double[] prevActivation = activations.get(i); // A_{i-1}

                delta = layer.backward(delta, prevActivation, learningRate);
                // delta è ora il gradiente rispetto all'input di questo layer,
                // cioè serve come errore per il layer precedente
            }
        }

        System.out.printf("Epoch %d - Loss: %.6f%n", epoch + 1, totalLoss / trainingSet.size());
    }
}
```

## Buone Combinazioni
Le funzioni di attivazione, le funzioni di loss e gli inizializzatori dei pesi non lavorano a comparti stagni. Una buona combinazione tra queste tre variabili può aiutare ad ottenere risultati migliori. Vediamo alcune combinazioni e per quale tipo di problema sono più indicate.

<img width="784" alt="image" src="https://github.com/user-attachments/assets/35164521-3cbe-4b86-aae5-e2d705c0a59e" />

Per il nostro caso d’uso, la combinazione migliore sarebbe la prima, però la Softmax ha un problema: non lavora sul singolo dato ma su un vettore, cioè va applicata all’intero layer e non al singolo neurone! Questo non solo rompe il contratto dell’interfaccia ActivationFunction ma ci costringe a inserire logiche speciali solo per l’AF. Per il momento proverei a non utilizzarla, misuriamo la qualità dei risultati ottenuti senza Softmax. 

Inizierei con queste impostazioni:

Rete
Funzione di loss	MSE
N° di epoche	10
Learning rate	0,01

Input layer
N° di neuroni 	128
Funzione di attivazione 	ReLU
Funzione di inizializzazione pesi	HE
	
Hidden layer 1
N° di neuroni 	64
Funzione di attivazione 	ReLU
Funzione di inizializzazione pesi	HE

Output layer  
N° di neuroni 	10
Funzione di attivazione 	Sigmoid
Funzione di inizializzazione pesi	Xavier

Vediamo che succede.

```
// lettura del campione di addestramento
//
long startTime = System.currentTimeMillis();

TrainReader tr = new TrainReader("src/main/resources/mnist_train.csv");
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

NeuralNetwork nn = new NeuralNetwork(DataGrid.getSize(), lossFunction);
nn.addLayer(128, hiddenInit, hiddenAf);
nn.addLayer(64, hiddenInit, hiddenAf);
nn.addLayer(10, outputInit, outputAf);      // output layer con 10 neuroni

System.out.println("Tempo impiegato per la creazione della rete neurale: " + (System.currentTimeMillis() - startTime) + " ms");


// addestramento della rete neurale
//
startTime = System.currentTimeMillis();
nn.train(tr.getDataGrids(), 10, 0.01);

System.out.println("Tempo impiegato per l'addestramento: " + (System.currentTimeMillis() - startTime) + " ms");
```

Ottengo questa risposta sulla mia macchina (senza usare né GPU né parallelismi):

```
Tempo impiegato per la lettura del campione: 3403 ms
Dimensione campione: 60000
Tempo impiegato per la creazione della rete neurale: 19 ms
Epoch 1 - Loss: 0,030135
Epoch 2 - Loss: 0,014432
Epoch 3 - Loss: 0,011771
Epoch 4 - Loss: 0,010216
Epoch 5 - Loss: 0,009136
Epoch 6 - Loss: 0,008297
Epoch 7 - Loss: 0,007623
Epoch 8 - Loss: 0,007059
Epoch 9 - Loss: 0,006568
Epoch 10 - Loss: 0,006140
Tempo impiegato per l'addestramento: 106339 ms
```

La progressione della loss è eccellente: decresce costantemente in 10 epoche, segno che la rete sta effettivamente apprendendo. Ci mette circa 10 secondi per epoca e decresce stabilmente, senza impennate: non c'è overfitting né instabilità.

Il valore finale della loss è molto buono per MSE su classificazione a 10 classi. Potrei aggiungere del semplice codice per calcolare anche l’accuratezza, ma questo calcolo inficia molto sulle prestazioni, aumentando i tempi di esecuzione. Giusto per curiosità ecco i risultati con l’accuratezza:

```
Tempo impiegato per la lettura del campione: 2841 ms
Dimensione campione: 60000
Tempo impiegato per la creazione della rete neurale: 13 ms
Epoch 1 - Loss: 0,029933 - Accuracy: 84,69%
Epoch 2 - Loss: 0,014414 - Accuracy: 93,21%
Epoch 3 - Loss: 0,011934 - Accuracy: 94,42%
Epoch 4 - Loss: 0,010429 - Accuracy: 95,18%
Epoch 5 - Loss: 0,009349 - Accuracy: 95,72%
Epoch 6 - Loss: 0,008526 - Accuracy: 96,17%
Epoch 7 - Loss: 0,007871 - Accuracy: 96,51%
Epoch 8 - Loss: 0,007324 - Accuracy: 96,79%
Epoch 9 - Loss: 0,006853 - Accuracy: 97,03%
Epoch 10 - Loss: 0,006438 - Accuracy: 97,23%
Tempo impiegato per l'addestramento: 183571 ms
```

Usiamo ora la CrossEntropy come funzione di perdita per vedere che differenze ci sono di accuratezza e progressione. Va prima implementata:

```
public class CrossEntropy implements LossFunction
{
    @Override
    public double compute(double[] prediction, double[] target)
    {
        double epsilon = 1e-12; // per evitare log(0)
        double loss = 0.0;
        for (int i = 0; i < prediction.length; i++)
            loss -= target[i] * Math.log(prediction[i] + epsilon);
        return loss;
    }

    @Override
    public double[] derivative(double[] prediction, double[] target)
    {
        // Derivata della softmax + cross-entropy semplificata
        double[] grad = new double[prediction.length];
        for (int i = 0; i < prediction.length; i++)
            grad[i] = prediction[i] - target[i];
        return grad;
    }
}
```

Ed eseguiamo l’apprendimento con questa funzione:

```
Tempo impiegato per la lettura del campione: 2372 ms
Dimensione campione: 60000
Tempo impiegato per la creazione della rete neurale: 16 ms
Epoch 1 - Loss: 15,093006 - Accuracy: 70,71%
Epoch 2 - Loss: 23,291681 - Accuracy: 53,93%
Epoch 3 - Loss: 24,262962 - Accuracy: 43,07%
Epoch 4 - Loss: 23,650481 - Accuracy: 44,07%
Epoch 5 - Loss: 23,704655 - Accuracy: 42,29%
Epoch 6 - Loss: 26,733997 - Accuracy: 25,30%
Epoch 7 - Loss: 25,745070 - Accuracy: 27,29%
Epoch 8 - Loss: 26,539806 - Accuracy: 21,36%
Epoch 9 - Loss: 27,629399 - Accuracy: 11,30%
Epoch 10 - Loss: 27,629399 - Accuracy: 11,30%
Tempo impiegato per l'addestramento: 184381 ms
```

Questo mostra che l’accuratezza crolla drasticamente con questa combinazione e che il loss aumenta invece di diminuire, sintomo di instabilità o divergenza. Sarebbe da provare la combinazione buona CrossEntropy + Softmax, ma come detto richiede una profonda revisione di tutta la rete.

## Salvare il lavoro
Dato che per addestrare la rete serve un tempo non trascurabile, mi piacerebbe salvare i risultati dell’addestramento (serializzarli) in un file, per poi ricaricarli senza eseguire tutto di nuovo. In questo modo si potrebbero anche far generare più file a seconda della combinazione di algoritmi usata. 

Per farlo basta rendere tutte le classi Serializable e prevedere due semplici metodi di scrittura e lettura:

```
void saveNetwork(String filename)
{
    try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename)))
    {
        out.writeObject(neuralNetwork);
    }
    catch (FileNotFoundException e)
    {
        throw new RuntimeException(e);
    } catch (IOException e)
    {
        throw new RuntimeException(e);
    }
}

NeuralNetwork loadNetwork(String filename)
{
    try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename)))
    {
        return (NeuralNetwork) in.readObject();
    }
    catch (FileNotFoundException e)
    {
        throw new RuntimeException(e);
    }
    catch (IOException | ClassNotFoundException e)
    {
        throw new RuntimeException(e);
    }
}
```

Ora basterà passare su riga di comando le proprie intenzioni, cioè se si vuole addestrare il sistema salvando il file risultante oppure se caricare un precedente addestramento. Nel main:

```
private void go(String[] args)
{
    if ( args.length != 1 )
        printUsage();

    String argument = args[0];
    if ( argument.startsWith("--output=") )
    {
        String filename = argument.substring("--output=".length());
        addestramento();
        saveNetwork(filename);
    }
    else if ( argument.startsWith("--input=") )
    {
        String filename = argument.substring("--input=".length());
        neuralNetwork = loadNetwork(filename);
        //… predizione qui
    }
    else
        printUsage();
}
```

## La rete predice
Dopo l’apprendimento è ora di mettere la rete al lavoro, fornendole in input un numero scritto a mano e verificando la sua predizione. 

Iniziamo con la predizione di un singolo numero, accettiamo quindi un input sulla linea di comando. Faremo copia e incolla di uno dei numeri presenti nel file di test. Il metodo deve prendere i numeri inseriti nel prompt e separati da spazi o da carattere a capo.

```
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

        int predicted = IntStream.range(0, output.length)
                .boxed()
                .max(comparingDouble(i -> output[i]))
                .orElse(-1);

        System.out.println("La rete neurale ha predetto: " + predicted);
    }
}
```

### SPAZIO NERD
È interessante notare il calcolo della previsione usando stream e lambda. L’obiettivo è trovare la classe con la probabilità più alta. Quando facciamo una previsione, la rete neurale restituisce un array di double (lunghezza 10 per MNIST), dove ogni valore rappresenta la “confidenza” della rete che l’input sia quella cifra. Ad esempio:

```
double[] output = nn.predict(input);
// output = [0.01, 0.05, 0.12, 0.02, 0.60, 0.03, 0.01, 0.10, 0.03, 0.03]
```

Vogliamo trovare l'indice del valore massimo, perché quell'indice rappresenta la cifra predetta. Dove: 

•	range crea una sequenza di interi da 0 a 9.
•	boxed serve per trasformare lo stream di primitivi interi in uno stream di oggetti Integer, lo facciamo perché Stream<Integer> permette l’uso di Comparator.comparing().
•	max confronta tutti gli indici in base al valore corrispondente in output, e restituisce l’indice del valore massimo. Nota: i -> output[i] dice al comparatore di usare il valore dell’output come criterio.
•	orElse nel caso quasi impossibile in cui non venga trovato il massimo, torna -1. 

Ma anche in questo caso l’eleganza penalizza un po’ le prestazioni. Per prestazioni massime il caro vecchio loop è la scelta migliore. 

Ora proviamo a darle un paio di input:

<img width="365" alt="image" src="https://github.com/user-attachments/assets/4c1de015-1988-4c54-ab71-c831ecdf7548" />

<img width="383" alt="image" src="https://github.com/user-attachments/assets/78a34113-ac96-42ee-afc4-5e7cec8e1cd6" />

<img width="363" alt="image" src="https://github.com/user-attachments/assets/13fa375e-bbb9-4e3e-992b-f9f529afd21d" />

<img width="378" alt="image" src="https://github.com/user-attachments/assets/4c1d8320-c2d8-409c-8a78-7aa2f11c74e9" />

## Un test strutturato
Proviamo a dare in pasto alla rete tutto il file di test da 10.000 righe e verificare l’accuratezza delle sue predizioni. Ci serve una riga di comando migliorata, con tre modalità operative:

.  Modalità di apprendimento, con salvataggio del file:
```java -jar JaNN.jar --mode=output --model=train1.nn --input=mnist_train.csv```

.  Modalità di lettura da riga di comando delle singole cifre da predire:
```java -jar JaNN.jar --mode=input --model=train1.nn```

.  Modalità di test, lettura delle cifre da un file grande:
```java -jar JaNN.jar --mode=test --model=train1.nn --input=mnist_test.csv```


Modifichiamo il Main:

```
private void go(String[] args)
{
    if ( args.length != 1 )
        printUsage();

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
```

Lanciamo il test, riporto solo i primi 20 risultati su 10.000 righe stampate e poi il resoconto finale:

#     Label   Predicted  Correct?   Time(ms)  
0     7       7          YES        7         
1     2       2          YES        3         
2     1       1          YES        3         
3     0       0          YES        2         
4     4       4          YES        2         
5     1       1          YES        2         
6     4       4          YES        2         
7     9       9          YES        2         
8     5       6          NO         2         
9     9       9          YES        2         
10    0       0          YES        2         
11    6       6          YES        2         
12    9       9          YES        2         
13    0       0          YES        2         
14    1       1          YES        2         
15    5       5          YES        2         
16    9       9          YES        2         
17    7       7          YES        2         
18    3       3          YES        2         
19    4       4          YES        2         
20    9       9          YES        2         
… omesse le altre

Accuracy: 96,34% (9634/10000)
Tempo medio: 2,14 ms per predizione

L’accuratezza è buona ma non perfetta. Cosa si può fare? Le reti fully connected, benché semplici da implementare, non sono le migliori per questo tipo di problema. Sarebbe invece più opportuno utilizzare una Convolutional Neural Network, CNN, che usa filtri che scorrono sull’immagine per estrarre pattern locali. E’ molto efficace sulle immagini ma un po’ più difficile da implementare. 

In questo caso il codice andrebbe rivisto profondamente perché la CNN non lavora con array monodimensionali ma con matrici o volumi (double[][] o double[][][]). 

## Conclusioni e prossimi passi
In questo progetto è stata implementata una rete neurale fully connected in Java, con addestramento tramite backpropagation e supporto per diverse funzioni di attivazione e inizializzazione dei pesi. Il sistema è stato testato sul dataset MNIST per il riconoscimento di cifre scritte a mano, con ottimi risultati: oltre il 96% di accuratezza su 10.000 campioni di test e tempi di inferenza inferiori ai 3 ms per predizione (sulla mia macchina).

Sono stati inoltre introdotti:
. il salvataggio e caricamento della rete su file,
. una modalità interattiva per la predizione da input manuale,
. e una modalità batch per il test su file CSV.

## Risultati principali
.  La rete fully connected con due hidden layer e attivazioni ReLU + Sigmoid si è dimostrata efficace per il problema.
.  L'uso della normalizzazione dei dati e di inizializzatori come He e Xavier ha migliorato significativamente la stabilità del training.
.  La funzione di loss MSE ha prodotto risultati solidi, mentre l’uso di cross-entropy richiede l’integrazione corretta della funzione Softmax, non effettuata.

## Sviluppi futuri
Per estendere le capacità e la precisione della rete, si potrebbero implementare i seguenti miglioramenti:

1. Funzione di attivazione Softmax
Implementare Softmax come funzione di attivazione per l’output layer, in combinazione con la loss function Cross-Entropy, per un comportamento più coerente nella classificazione multi-classe.

2. Supporto per reti convoluzionali (CNN)
Introdurre layer di convoluzione e pooling, in modo da:
	. ridurre il numero di parametri,
	. sfruttare la struttura spaziale delle immagini,
	. e migliorare l’accuratezza  

3. Confusion Matrix e metriche avanzate
Aggiungere la stampa di una matrice di confusione e il calcolo di precision, recall e F1-score per una valutazione più approfondita del comportamento della rete.

4. Visualizzazione dell’addestramento
Mostrare graficamente l’andamento della loss e dell’accuratezza nel tempo, ad esempio salvando un CSV per ogni epoca o integrando una piccola GUI.

5. Interfaccia grafica per l’inferenza
Creare una semplice GUI che permetta all’utente di disegnare a mano una cifra e ricevere la predizione in tempo reale.
