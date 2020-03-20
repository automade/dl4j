import java.io.File
import java.net.URL

import org.apache.commons.io.FileUtils

object DatasetDownloader {

  private val DATASETS: Map[String, String] = Map(
    "iris" -> "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
  )

}

class DatasetDownloader {

  def downloadDataset(dataset: String, datasetPath: String): Unit = {
    val datasetDir: File = new File(s"$datasetPath/$dataset.data")
    if (!datasetDir.getParentFile.exists()) {
      datasetDir.getParentFile.mkdirs()
    }
    if (!datasetDir.exists()) {
      val url: URL = new URL(DatasetDownloader.DATASETS(dataset))
      println(s"Downloading $dataset dataset from ${url.toString}")
      FileUtils.copyURLToFile(url, datasetDir)
    }
  }

}
