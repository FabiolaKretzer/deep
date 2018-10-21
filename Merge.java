package merge;

import merge.Algorithms;
import weka.classifiers.Evaluation;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Merge {
    
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("merge/weather.numeric.arff");
        Instances dataset = source.getDataSet();
        
        Algorithms alg = new Algorithms();
        Instances newData = alg.feature_selection(dataset);
        
        System.out.println(newData);
        
        Evaluation eval = alg.deep_learning(newData);
        
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        
        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Incorret % = " + eval.pctIncorrect());
        System.out.println("AUC = " + eval.areaUnderROC(1));
        System.out.println("kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precision =" + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        
        System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));   
    }
    
}
