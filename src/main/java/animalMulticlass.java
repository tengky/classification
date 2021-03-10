import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class animalMulticlass {

    static int seed = 123;
    static int numInput = 16;
    static int numClass = 7;
    static int epoch = 500;
    static double splitRatio = 0.8;
    static double learningRate = 1e-3;

    public static void main(String[] args) throws Exception {

        //set filepath
        File datafile = new ClassPathResource("/animals_multiclass/zoo.csv").getFile();

        //set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(datafile));

        //Build schema to prepare data
        Schema sc = new Schema.Builder()
                .addColumnCategorical("animal_name", Arrays.asList(
                        "aardvark",
                        "antelope",
                        "bass",
                        "bear",
                        "boar",
                        "buffalo",
                        "calf",
                        "carp",
                        "catfish",
                        "cavy",
                        "cheetah",
                        "chicken",
                        "chub",
                        "clam",
                        "crab",
                        "crayfish",
                        "crow",
                        "deer",
                        "dogfish",
                        "dolphin",
                        "dove",
                        "duck",
                        "elephant",
                        "flamingo",
                        "flea",
                        "frog",
                        "fruitbat",
                        "giraffe",
                        "girl",
                        "gnat",
                        "goat",
                        "gorilla",
                        "gull",
                        "haddock",
                        "hamster",
                        "hare",
                        "hawk",
                        "herring",
                        "honeybee",
                        "housefly",
                        "kiwi",
                        "ladybird",
                        "lark",
                        "leopard",
                        "lion",
                        "lobster",
                        "lynx",
                        "mink",
                        "mole",
                        "mongoose",
                        "moth",
                        "newt",
                        "octopus",
                        "opossum",
                        "oryx",
                        "ostrich",
                        "parakeet",
                        "penguin",
                        "pheasant",
                        "pike",
                        "piranha",
                        "pitviper",
                        "platypus",
                        "polecat",
                        "pony",
                        "porpoise",
                        "puma",
                        "pussycat",
                        "raccoon",
                        "reindeer",
                        "rhea",
                        "scorpion",
                        "seahorse",
                        "seal",
                        "sealion",
                        "seasnake",
                        "seawasp",
                        "skimmer",
                        "skua",
                        "slowworm",
                        "slug",
                        "sole",
                        "sparrow",
                        "squirrel",
                        "starfish",
                        "stingray",
                        "swan",
                        "termite",
                        "toad",
                        "tortoise",
                        "tuatara",
                        "tuna",
                        "vampire",
                        "vole",
                        "vulture",
                        "wallaby",
                        "wasp",
                        "wolf",
                        "worm",
                        "wren"
                ))
                .addColumnsInteger("hair" ,"feathers" ,"eggs", "milk", "airborne", "aquatic", "predator", "toothed" ,"backbone" ,"breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "class_type")
                .build();

        //Build transform process
        TransformProcess tp = new TransformProcess.Builder(sc)
                .removeColumns("animal_name")
                .build();

        Schema outputSchema = tp.getFinalSchema();
        System.out.println(outputSchema);

        List<List<Writable>> allData = new ArrayList<>();

        while (rr.hasNext()) {
            allData.add(rr.next());
        }
        List<List<Writable>> processData = LocalTransformExecutor.execute(allData, tp);

        //Create iterator from process data
        CollectionRecordReader collectionRR = new CollectionRecordReader(processData);

        //Input batch size, label index and number of label
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectionRR, processData.size(), -1, 7);

        //Create Iterator and shuffle the dataset
        DataSet fullDataset = dataSetIterator.next();
        fullDataset.shuffle(seed);

        //Input spilt ratio
        SplitTestAndTrain testAndTrain = fullDataset.splitTestAndTrain(splitRatio);

        //Get train and test dataset
        DataSet trainData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //printout size
        System.out.println("Training vector: ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        //Data normalization
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);

        //Build network configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(numClass)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        //Define network
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        //UI-Evaluator
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //Set model listeners
        model.setListeners(new StatsListener(storage, 50));

        //Training
        Evaluation eval;
        for(int i=0; i < epoch; i++) {
            model.fit(trainData);
            eval = model.evaluate(new ViewIterator(testData, processData.size()));
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }

        //Confusion matrix
        Evaluation evalTrain = model.evaluate(new ViewIterator(trainData, processData.size()));
        Evaluation evalTest = model.evaluate(new ViewIterator(testData,processData.size()));
        System.out.print("Train Data");
        System.out.println(evalTrain.stats());

        System.out.print("Test Data");
        System.out.print(evalTest.stats());
    }
}
