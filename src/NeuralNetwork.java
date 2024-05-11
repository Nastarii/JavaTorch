import java.util.List;

public class NeuralNetwork {
    
    private List<Layer> layers;
    private LossFunction lossFunction;
    private Optimizer optimizer;
    private int epochs;

    public NeuralNetwork(LossFunction lossFunction, Optimizer optimizer, int epochs) {
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.epochs = epochs;
    }

    public void Sequential(Layer... layers) {
        for (Layer layer : layers) {
            this.layers.add(layer);
        }
    }

    public double[] forward(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

    public double[] getWeights() {
        double[] weights = new double[0];
        for (Layer layer : layers) {
            double[] layerWeights = layer.getWeights();
            double[] newWeights = new double[weights.length + layerWeights.length];
            System.arraycopy(weights, 0, newWeights, 0, weights.length);
            System.arraycopy(layerWeights, 0, newWeights, weights.length, layerWeights.length);
            weights = newWeights;
        }
        return weights;
    }
    
    public void train(double[] inputs, double target) {
        for (int i = 0; i < epochs; i++) {
            double[] outputs = forward(inputs);
            double loss = lossFunction.calculate(outputs[outputs.length - 1], target);
            this.optimizer.backward(getWeights(), loss);
            System.out.println("Epoch: " + i + " Output: " + outputs[outputs.length - 1]);
        }
    }
}
