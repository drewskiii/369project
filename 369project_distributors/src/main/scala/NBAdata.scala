import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object NBAdata{
  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local")
      .appName("369project")
      .getOrCreate()

    val schema = StructType(
      StructField("Rank", StringType, nullable = true) ::
        StructField("Name", StringType, nullable = true) ::
        StructField("Team", StringType, nullable = true) ::
        StructField("Position", StringType, nullable = true) ::
        StructField("Age", StringType, nullable = true) ::
        StructField("GP", StringType, nullable = true) ::
        StructField("MPG", StringType, nullable = true) ::
        StructField("MIN%", StringType, nullable = true) ::
        StructField("USG%", DoubleType, nullable = true) ::
        StructField("TO%", StringType, nullable = true) ::
        StructField("FTA", StringType, nullable = true) ::
        StructField("FT%", StringType, nullable = true) ::
        StructField("2PA", StringType, nullable = true) ::
        StructField("2P%", StringType, nullable = true) ::
        StructField("3PA", StringType, nullable = true) ::
        StructField("3P%", StringType, nullable = true) ::
        StructField("eFG%", StringType, nullable = true) ::
        StructField("TS%", StringType, nullable = true) ::
        StructField("PPG", StringType, nullable = true) ::
        StructField("RPG", StringType, nullable = true) ::
        StructField("TRB%", StringType, nullable = true) ::
        StructField("APG", DoubleType, nullable = true) ::
        StructField("AST%", StringType, nullable = true) ::
        StructField("SPG", StringType, nullable = true) ::
        StructField("BPG", StringType, nullable = true) ::
        StructField("TOPG", StringType, nullable = true) ::
        StructField("VI", StringType, nullable = true) ::
        StructField("ORTG", StringType, nullable = true) ::
        StructField("DRTG", StringType, nullable = true) ::
        Nil)

    val NBADF = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ",")
      .option("mode", "DROPMALFORMED")
      //.option("timestampFormat", "yyyy/MM/dd HH:mm:ss")
      .schema(schema)
      .load(getClass.getResource("NBA.csv").getPath)
      .cache()

    val df =  NBADF.drop("Age", "GP", "MPG", "MIN%", "TO%", "FTA", "FT%", "2PA", "2P%", "3PA", "3P%", "ORTG",
    "DRTG", "VI", "TOPG", "SPG", "BPG", "AST%", "TRB%", "eFG%", "RPG", "TS%", "Rank",
      "PPG").filter("Team == 'San'")

    // transform userDf with VectorAssembler to add feature column
    val cols = Array("USG%", "APG")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df)

    featureDf.show()

    // kmeans model with 3 clusters
    val kmeans = new KMeans()
      .setK(3)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    val kmeansModel = kmeans.fit(featureDf)
    kmeansModel.clusterCenters.foreach(println)

    // test the model with test data set
    val predictDf = kmeansModel.transform(featureDf)
    predictDf.show()

    // no of categories
    predictDf.groupBy("prediction").count().show()
  }
}