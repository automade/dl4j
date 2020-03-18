import sbt._

object Dependencies {

  lazy val dl4jVersion = "1.0.0-beta6"

  val cuda = "org.nd4j" % "nd4j-cuda-10.2" % dl4jVersion
  val dl4j = "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion


  val deps = Seq(cuda, dl4j)

}