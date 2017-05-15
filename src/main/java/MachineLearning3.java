import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class MachineLearning3 {

    public static final int NUMBER_OF_DATA_SETS = 40000;
    public static final int NUMBER_OF_COLUMNS = 24;
    private static final int NUM_HIDDEN_NODES = 125;
    private static final int TRAINING_TIME = 1000;
    private static final int MAX_EPOCHS = 20;
    private static final String LEARN_FILE_PATH = "src\\main\\resources\\learnUIC.csv";
    private static final String EVAL_FILE_PATH = "src\\main\\resources\\evalUIC.csv";

    public static void main(String[] args) {

        evalCVS();

        List<List<String>> data = getDataAsList();
        data = normalize(data);
        networkLearn(data);

        List<List<String>> dataEval = getEvalDataAsList();
        dataEval = normalizeEvalData(dataEval);
        evaluateNetwork(dataEval);
    }

    private static void networkLearn(List<List<String>> data) {
        int seed = 123;
        int batchSize = 50;
        int numInputs = 23;
        int numOutputs = 2;
        RecordReader rr = new ListStringRecordReader();

        try {
            rr.initialize(new ListStringSplit(data));
        } catch (Exception e) {
            System.out.println(e);
        }
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 23, 2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAGRAD)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(numOutputs)
                        .build()
                )
                .pretrain(false).backprop(true).build();

        // Apply Network and attach Listener to Web-UI
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage));
        uiServer.attach(statsStorage);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(MAX_EPOCHS))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(TRAINING_TIME, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(trainIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning3\\"))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        File locationToSave = new File("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning3\\NeuralNetwork.zip");
        boolean saveUpdater = true;

        org.deeplearning4j.nn.api.Model model2 = result.getBestModel();

        try {
            ModelSerializer.writeModel(model2, locationToSave, saveUpdater);
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    private static List<List<String>> getDataAsList() {

        ArrayList<List<String>> data = null;
        try {
            String fileName = LEARN_FILE_PATH;
            BufferedReader br = null;
            String sCurrentLine;
            br = new BufferedReader(new FileReader(fileName));//file name with path
            data = new ArrayList<List<String>>();

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
    private static void evalCVS() {


        try {
            String fileName = LEARN_FILE_PATH;
            BufferedReader br = null;
            String sCurrentLine;
            br = new BufferedReader(new FileReader(fileName));//file name with path
            String [] parts1 = new String [23];
            int x = 0;

            while ((sCurrentLine = br.readLine()) != null) {
                for (int i = 0; i < sCurrentLine.length(); i++){
                    char c = sCurrentLine.charAt(i);
                    if(c == ','){
                        x++;
                    }
                }
                if(x != 23){
                    System.out.println("Problem: " + sCurrentLine);
                }
                x = 0;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static List<List<String>> normalize(List<List<String>> data) {

        double[] values = new double[NUMBER_OF_DATA_SETS];
        double[] mins = new double[NUMBER_OF_COLUMNS];
        double[] maxs = new double[NUMBER_OF_COLUMNS];

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < data.size(); i++) {
                    values[i] = Double.parseDouble(data.get(i).get(j));
            }
            mins[j] = Arrays.stream(values).min().getAsDouble();
            maxs[j] = Arrays.stream(values).max().getAsDouble();
        }
        for (List<String> temp : data) {
            for (int y = 0; y < NUMBER_OF_COLUMNS; y++) {
                if (mins[y] != maxs[y]) {
                    double tempValue = (Double.parseDouble(temp.get(y)) - mins[y]) / (maxs[y] - mins[y]);

                    if (tempValue == 0.0) {
                        temp.set(y, "0");
                    } else if (tempValue == 1.0) {
                        temp.set(y, "1");
                    } else {
                        temp.set(y, Double.toString(tempValue));
                    }
                }
            }
        }
        return data;
    }

    private static List<List<String>> getEvalDataAsList() {

        ArrayList<List<String>> data = null;
        try {
            String fileName = EVAL_FILE_PATH;
            BufferedReader br = null;
            String sCurrentLine;
            br = new BufferedReader(new FileReader(fileName));//file name with path
            data = new ArrayList<List<String>>();

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static List<List<String>> normalizeEvalData(List<List<String>> data) {
        double[] values = new double[NUMBER_OF_DATA_SETS];
        double[] mins = new double[NUMBER_OF_COLUMNS];
        double[] maxs = new double[NUMBER_OF_COLUMNS];
        int counterDefaults = 1;
        int counterNonDefaults = 1;
        List<List<String>> data2 = new ArrayList<List<String>>();

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < data.size(); i++) {
                values[i] = Double.parseDouble(data.get(i).get(j));
            }
            mins[j] = Arrays.stream(values).min().getAsDouble();
            maxs[j] = Arrays.stream(values).max().getAsDouble();
        }
        for (List<String> temp : data) {
            for (int y = 0; y < NUMBER_OF_COLUMNS; y++) {
                if (mins[y] != maxs[y]) {
                    double tempValue = (Double.parseDouble(temp.get(y)) - mins[y]) / (maxs[y] - mins[y]);

                    if (tempValue == 0.0) {
                        temp.set(y, "0");
                    } else if (tempValue == 1.0) {
                        temp.set(y, "1");
                    } else {
                        temp.set(y, Double.toString(tempValue));
                    }
                }
            }
            data2.add(temp);
        }
        return data2;
    }

    private static void evaluateNetwork(List<List<String>> dataEval) {
        RecordReader rrTest = new ListStringRecordReader();
        MultiLayerNetwork model = null;
        int batchSize = 10;
        int numOutputs = 2;

        try {
            rrTest.initialize(new ListStringSplit(dataEval));
        } catch (Exception e) {
            System.out.println(e);
        }

        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 23, 2);

        try {
            model = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning3\\NeuralNetwork.zip");
        } catch (Exception e) {
            System.out.println(e.toString());
        }

        System.out.println("Evaluate model.......");
        Evaluation eval = new Evaluation(numOutputs);

        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(lables, predicted);
        }
        System.out.println(eval.stats());
    }
}
