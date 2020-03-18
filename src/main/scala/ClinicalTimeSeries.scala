import java.io._
import java.net.URL

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.ROC
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions


object ClinicalTimeSeries {

  def downloadData(dataPath: String): Unit = {
    val DATA_URL = "https://dl4jdata.blob.core.windows.net/training/physionet2012/physionet2012.tar.gz"
    val directory = new File(dataPath)

    if (directory.exists()) {
      println(s"Data directory already exists: $dataPath")
      return
    }

    println(s"Data directory: $dataPath")

    directory.mkdir() // create new directory at specified path

    val archizePath = dataPath + "physionet2012.tar.gz" // set path for tar.gz file
    val archiveFile = new File(archizePath) // create tar.gz file
    val extractedPath = dataPath + "physionet2012"
    val extractedFile = new File(extractedPath)

    FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile) // copy data from URL to file

    var fileCount = 0
    var dirCount = 0
    val BUFFER_SIZE = 4096

    val tais = new TarArchiveInputStream(new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(archizePath))))

    var entry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]

    while (entry != null) {
      if (entry.isDirectory) {
        new File(dataPath + entry.getName).mkdirs()
        dirCount = dirCount + 1
        fileCount = 0
      }
      else {

        val data = new Array[scala.Byte](4 * BUFFER_SIZE)

        val fos = new FileOutputStream(dataPath + entry.getName)
        val dest = new BufferedOutputStream(fos, BUFFER_SIZE)
        var count = tais.read(data, 0, BUFFER_SIZE)

        while (count != -1) {
          dest.write(data, 0, count)
          count = tais.read(data, 0, BUFFER_SIZE)
        }

        dest.close()
        fileCount = fileCount + 1
      }
      if (fileCount % 1000 == 0) {
        print(".")
      }

      entry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]
    }
  }

  def getDataSetIterator(dataPath: String): (DataSetIterator, DataSetIterator) = {
    val NB_TRAIN_EXAMPLES = 3200 // number of training examples
    val NB_TEST_EXAMPLES = 800 // number of testing examples

    val path = FilenameUtils.concat(dataPath, "physionet2012/") // set parent directory

    val featureBaseDir = FilenameUtils.concat(path, "sequence") // set feature directory
    val mortalityBaseDir = FilenameUtils.concat(path, "mortality") // set label directory

    // Load training data

    val trainFeatures = new CSVSequenceRecordReader(1, ",")
    trainFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1))

    val trainLabels = new CSVSequenceRecordReader()
    trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1))

    val trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, 32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)


    // Load testing data
    val testFeatures = new CSVSequenceRecordReader(1, ",")
    testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1))

    val testLabels = new CSVSequenceRecordReader()
    testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1))

    val testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
      32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

    (trainData, testData)
  }

  def getModel: ComputationGraph = {
    val NB_INPUTS = 86
    val NB_EPOCHS = 10
    val RANDOM_SEED = 1234
    val LEARNING_RATE = 0.005
    val BATCH_SIZE = 32
    val LSTM_LAYER_SIZE = 200
    val NUM_LABEL_CLASSES = 2

    val conf = new NeuralNetConfiguration.Builder()
      .seed(RANDOM_SEED)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(LEARNING_RATE))
      .weightInit(WeightInit.XAVIER)
      .dropOut(0.25)
      .graphBuilder()
      .addInputs("trainFeatures")
      .setOutputs("predictMortality")
      .addLayer("L1", new GravesLSTM.Builder()
        .nIn(NB_INPUTS)
        .nOut(LSTM_LAYER_SIZE)
        .forgetGateBiasInit(1)
        .activation(Activation.TANH)
        .build(),
        "trainFeatures")
//      .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
        .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
        .nIn(LSTM_LAYER_SIZE).nOut(NUM_LABEL_CLASSES).build(), "L1")
      .build()

    val model = new ComputationGraph(conf)
    model
  }

  def main(args: Array[String]): Unit = {
    val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_physionet/")

    downloadData(DATA_PATH)
    val (trainData, testData) = getDataSetIterator(DATA_PATH)

    val model: ComputationGraph = getModel

    model.fit(trainData, 2)

    val roc = new ROC(100)

    while (testData.hasNext) {
      val batch = testData.next()
      val output = model.output(batch.getFeatures)
      roc.evalTimeSeries(batch.getLabels, output(0))
    }

    println("FINAL TEST AUC: " + roc.calculateAUC());
  }

}
