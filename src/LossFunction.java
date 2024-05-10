public class LossFunction {
    
    private String lossFunction;

    public LossFunction(String lossFunction) {
        this.lossFunction = lossFunction;
        if (!lossFunction.equals("MSE") && 
            !lossFunction.equals("MAE") && 
            !lossFunction.equals("CrossEntropy")
        ) {
            throw new IllegalArgumentException("Invalid loss function");
        }  
    }

    public double MSE(double predicted, double target) {
        return Math.pow(predicted - target, 2);
    }

    public double MAE(double predicted, double target) {
        return Math.abs(predicted - target);
    }

    public double CrossEntropy(double predicted, double target) {
        return -target * Math.log(predicted) - (1 - target) * Math.log(1 - predicted);
    }

    public double calculate(double predicted, double target) {
        switch (this.lossFunction) {
            case "MSE":
                return MSE(predicted, target);
            case "MAE":
                return MAE(predicted, target);
            case "CrossEntropy":
                return CrossEntropy(predicted, target);
            default:
                throw new IllegalArgumentException("Invalid loss function");
        }
    }
}
