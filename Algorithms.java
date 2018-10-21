package merge;

import java.util.Random;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Algorithms {
    
    Algorithms() {}
    
    public Instances feature_selection (Instances dataset) throws Exception {
        AttributeSelection filter = new AttributeSelection();
        
        CfsSubsetEval cfs = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        
        // search.setSearchBackwards(true);
        
        filter.setEvaluator(cfs);
        filter.setSearch(search);
        
        filter.setInputFormat(dataset);
        
        Instances newData = Filter.useFilter(dataset, filter);
        
        return newData;
    }
    
    public Evaluation deep_learning (Instances dataset) throws Exception {
        dataset.setClassIndex(1);
        
        MultilayerPerceptron mult = new MultilayerPerceptron();
        mult.buildClassifier(dataset);
        
        Evaluation eval = new Evaluation(dataset);
        
        Random rand = new Random(1);
        int folds = 10;
        
        eval.crossValidateModel(mult, dataset, folds, rand);
        
        return eval;
    }
}
