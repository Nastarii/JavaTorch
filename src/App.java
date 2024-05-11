public class App {
    public static void main(String[] args) throws Exception {
        
        Activation activation = new Activation("sigmoid"); // Define the activation function

        Optimizer optimizer = new Optimizer(0.32, 0.9, 0, false);// Define the optimizer

        LossFunction lossFunction = new LossFunction("MAE"); // Define the loss function

        Layer layer = new Layer(4, 4, activation); // Initialize a Layer with 4 inputs and 1 neuron
        Neuron neuron = new Neuron(4);  // Initialize a Neuron with 4 inputs

        double[] inputs = {0.93, 0.95, 0.97, 0.98};  // Input values

        double target = 0.9; // Target value to predict

        int epochs = 100; // Number of iterations of the training loop

        for (int i = 0; i < epochs; i++) {

            double[] outputs = layer.forward(inputs); // Calculate the output of the layer
            double output = neuron.forward(outputs, activation);  // Calculate the output of the neuron

            double loss = lossFunction.calculate(output, target);  // Calculate the loss between the predicted and target values

            double[] newLayerWeights = optimizer.backward(layer.getWeights(), loss); // Update optimizer
            layer.step(newLayerWeights, optimizer.getLearningRate() * loss); // Update weigths and bias

            double[] newWeights = optimizer.backward(neuron.getWeights(), loss); // Update optimizer
            neuron.step(newWeights, optimizer.getLearningRate() * loss); // Update weigths and bias

            System.out.println("Epoca:" + i + " Saida do neuronio: " + output); // Print Predictions
        }

    }
}
