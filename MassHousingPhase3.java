/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Ashuthosh
 */

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.lazy.IBk;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import weka.filters.Filter;
import weka.filters.supervised.attribute.ClassConditionalProbabilities;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.SortLabels;
import weka.filters.unsupervised.attribute.StringToNominal;


public class MassHousingPhase3 {
    
    //Loading Dataset from  csv file
    private Instances loadingData(String src) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(src)); // Loading the data from current directory
        Instances dataset = loader.getDataSet();
        
        return dataset;
    }
    
    //Placing the class attribute in the last
    private Instances sortLabels(Instances data_old) throws Exception{

        SortLabels sort = new SortLabels();
        Reorder order = new Reorder();
        order.setAttributeIndices("2-last,1");
        sort.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, sort);
        order.setInputFormat(data_new);
        data_new = Filter.useFilter(data_new, order);

        return data_new; // returns the sorted data placed the class attribute at last position
    }
    
    //Remove the attribut which are not required for classification
    private Instances removeAtt(Instances data_old) throws Exception{

        // remove attributes in a dataset to speed up the pre-processing.
        Remove remove =new Remove();
        remove.setAttributeIndices("1-7,9,10,15,18,20-52");
        remove.setInputFormat(data_old);
        Instances data_new = Filter.useFilter(data_old, remove);
        
        return data_new; 
    }
    
    //Normalize the data to make it more mean meaningful data
    private Instances normalizeData(Instances data)throws Exception{
        
        Normalize norm = new Normalize();
        norm.setScale(1.0);
        norm.setTranslation(0.0);
        norm.setInputFormat(data);
        Instances data_new = Filter.useFilter(data, norm);

        return data_new;
    }
    
    //Covert String type attribute to nominal since KNN takes numeric or nominal data for classification
    private Instances string2nominal(Instances data)throws Exception{
        StringToNominal sn = new StringToNominal();
        sn.setAttributeRange("first-last");
        sn.setInputFormat(data);
        data = Filter.useFilter(data, sn);
        
        return data;
    }
    
    //Printing stmt_date, rm_key, expected letter grade and predicted letter grade
    private void printResult(Evaluation validation, Instances data) throws Exception{
        ArrayList <Prediction> pred = validation.predictions(); // stores all the prediction of the data
        int count = 0;
        double num_key1[] = data.attributeToDoubleArray(0);
        System.out.println("False Predictions:");
        System.out.println("0:A and 1:Bad(D,F)");
        System.out.println("stmt_date \t rm_key \t expected \t predicted");
        // Check which all prediction were false and printing those instances stmt_data, rm_key, expected and predicted
        for(int k=0; k<data.numInstances(); k++){
            if(pred.get(k).predicted()!= pred.get(k).actual()){
                System.out.println(data.get(k).toString(6) + "\t  " + (int)num_key1[k] + "\t\t  " + (int)pred.get(k).actual() + "   \t\t   " + (int)pred.get(k).predicted());
                count++;
            }
        }
        //printing total incorrect classification for justification of printed instances
        System.out.println("total incorrect: " + count);
        //prints the report of the classification
        System.out.println(validation.toSummaryString(false));
        //prints the confusion matrix.
        //System.out.println(validation.toMatrixString());
    }
    
    /* ClassConditionalProbabilities creates five new attributes for each feature 
    and it converts nominal attributes with distinct values into something 
    more manageable for learning schemes that can handle nominal attributes*/
    private Instances classConditional(Instances data)throws Exception{
        //StringToNominal sn = new StringToNominal();
        ClassConditionalProbabilities c = new ClassConditionalProbabilities();
        data.setClassIndex(data.numAttributes()-1);
        c.setNominalConversionThreshold(-1); //So all attributes are used by classConditionalProbabilities
        c.setInputFormat(data);
        data = Filter.useFilter(data, c);
        
        return data;
    }
    
    public static void main(String[] args) throws IOException, Exception {
        MassHousingPhase3 m = new MassHousingPhase3();
        
        //Reading from the train file created by removing rm key present in the test and apply filters
        String src = "massnewtraindata(3545).csv";
        Instances newTrainData = m.loadingData(src);
        newTrainData = m.removeAtt(newTrainData);
        newTrainData = m.sortLabels(newTrainData);
        newTrainData = m.string2nominal(newTrainData);
        newTrainData = m.normalizeData(newTrainData);
        newTrainData = m.classConditional(newTrainData);
        
        //Reading from the extracted samples of the dataset and apply filters
        String src2 = "testdata(885).csv";
        Instances testNewData = m.loadingData(src2);
        Instances data2 = testNewData;
        testNewData = m.removeAtt(testNewData);
        testNewData = m.sortLabels(testNewData);
        testNewData = m.string2nominal(testNewData);
        testNewData = m.normalizeData(testNewData);
        testNewData = m.classConditional(testNewData);
        
        //Reading csv file of the new dataset provided and apply filters
        String src3 = "Test_Set_New_rm_key.csv";
        Instances testData = m.loadingData(src3);
        Instances data3 = testData;
        testData = m.removeAtt(testData);
        testData = m.sortLabels(testData);
        testData = m.string2nominal(testData);
        testData = m.normalizeData(testData);
        testData = m.classConditional(testData);
        
        //Reading from the original dataset provided to us and apply filters
        String src4 = "MassHousingTrainData.csv";
        Instances train = m.loadingData(src4);
        train = m.removeAtt(train);
        train = m.sortLabels(train);
        train = m.string2nominal(train);
        train = m.normalizeData(train);
        train = m.classConditional(train);
        
        //Preforming validation on extracted samples
        testNewData.setClassIndex(testNewData.numAttributes()-1);
        newTrainData.setClassIndex(newTrainData.numAttributes()-1);
        Classifier ibk_test = new IBk(3);
        Evaluation validation_new = new Evaluation(newTrainData);
        ibk_test.buildClassifier(newTrainData);
        validation_new.evaluateModel(ibk_test, testNewData);
        
        m.printResult(validation_new, data2);
        
        System.out.println("Now performing Testing on the given extra dataset: ");
        
        //Performing testing on the extra dataset
        testData.setClassIndex(testData.numAttributes()-1);
        train.setClassIndex(train.numAttributes()-1);
        Classifier ibk_final_test = new IBk(3);
        Evaluation validation = new Evaluation(train);
        ibk_final_test.buildClassifier(train);
        validation.evaluateModel(ibk_final_test, testData);
        
        m.printResult(validation, data3);

    }
}
 