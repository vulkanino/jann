package data;

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
