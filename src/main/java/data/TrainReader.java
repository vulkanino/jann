package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.System.exit;

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
