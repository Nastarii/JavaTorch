public class App {
    public static void main(String[] args) throws Exception {
        
        Activation activation = new Activation("sigmoid"); // Define the activation function

        Optimizer optimizer = new Optimizer(0.32, 0.9, 0, false);// Define the optimizer

        LossFunction lossFunction = new LossFunction("MAE"); // Define the loss function

        NeuralNetwork nn = new NeuralNetwork(lossFunction, optimizer, 100); // Define the Neural Network

        double[] inputs = {0.93, 0.95, 0.97, 0.98};  // Input values

        double target = 0.9; // Target value to predict

        nn.Sequential(
            new Layer(4, 4, activation), // Input Layer
            new Layer(1, 4, activation) // Output Layer
        );
        
        nn.train(inputs, target); // Train the Neural Network
    }
}
