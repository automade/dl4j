import Dependencies._

name := "dl4j"

version := "0.1"

scalaVersion := "2.13.1"

lazy val hello = (project in file("."))
  .settings(
    name := "dl4j",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % Test,
    libraryDependencies ++= deps,
  )