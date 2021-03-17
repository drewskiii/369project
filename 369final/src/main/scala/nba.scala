import org.apache.spark.{SparkConf, SparkContext, mllib};
//import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}



import java.io._
import scala.collection._
import scala.collection.immutable._
import scala.io._

object nba {
  def main(args: Array[String]): Unit = {

//    val conf = new SparkConf().setMaster("local[*]").setAppName("nba");
//    val sc = new SparkContext(conf);

//    val rdd = sc.parallelize(Array(5, 30, 10));
//    println(rdd.reduce(_+_));

//    val lines = sc.textFile("games_details.csv");

//    val starting_players = lines.take(2);
//    starting_players.foreach(println(_));

    val spark = SparkSession.builder
      .master("local")
      .appName("369final")
      .getOrCreate()

    val schema = StructType(
        StructField("GAME_ID", StringType, nullable = true) ::
        StructField("TEAM_ID", StringType, nullable = true) ::
        StructField("TEAM_ABBREVIATION", StringType, nullable = true) ::
        StructField("TEAM_CITY", StringType, nullable = true) ::
        StructField("PLAYER_ID", StringType, nullable = true) ::
        StructField("PLAYER_NAME", StringType, nullable = true) ::
        StructField("START_POSITION", StringType, nullable = true) ::
        StructField("COMMENT", StringType, nullable = true) ::
        StructField("MIN", StringType, nullable = true) ::
        StructField("FGM", DoubleType, nullable = true) ::
        StructField("FGA", DoubleType, nullable = true) ::
        StructField("FG_PCT", DoubleType, nullable = true) ::
        StructField("FG3M", DoubleType, nullable = true) ::
        StructField("FG3A", DoubleType, nullable = true) ::
        StructField("FG3_PCT", DoubleType, nullable = true) ::
        StructField("FTM", DoubleType, nullable = true) ::
        StructField("FTA", DoubleType, nullable = true) ::
        StructField("FT_PCT", DoubleType, nullable = true) ::
        StructField("OREB", DoubleType, nullable = true) ::
        StructField("DREB", DoubleType, nullable = true) ::
        StructField("REB", DoubleType, nullable = true) ::
        StructField("AST", DoubleType, nullable = true) ::
        StructField("STL", DoubleType, nullable = true) ::
        StructField("BLK", DoubleType, nullable = true) ::
        StructField("TO", DoubleType, nullable = true) ::
        StructField("PF", DoubleType, nullable = true) ::
        StructField("PTS", DoubleType, nullable = true) ::
        StructField("PLUS_MINUS", DoubleType, nullable = true) ::
        Nil)

    val NBADF = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ",")
      .option("mode", "DROPMALFORMED")
      //.option("timestampFormat", "yyyy/MM/dd HH:mm:ss")
      .schema(schema)
      .load("games_details.csv")
      .cache()

    NBADF.show();


    val df =  NBADF.drop("GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_ID", "PLAYER_NAME", "COMMENT", "FGM", "FGA", "FG3M", "FG3A",
      "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "STL", "BLK", "TO", "PF", "PLUS_MINUS")
        .filter("START_POSITION is not null and REB is not null and AST is not null and PTS is not null and MIN is not null");
    println("TEST");

    df.show();
    val cols = Array("REB", "AST", "PTS");
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df)

    featureDf.show();
//    df.collect().foreach(println(_));

    val kmeans = new KMeans()
      .setK(3)
      .setFeaturesCol("features")
      .setPredictionCol("prediction");
    // train model based on data of starting players
    val kmeansModel = kmeans.fit(featureDf);
    println("Kmeans Model")
    kmeansModel.clusterCenters.foreach(println);

    // test the model with test data set, using starting players again
     val predictDf = kmeansModel.transform(featureDf);

    println("HERE")
    predictDf.show();

    // no of categories
    predictDf.groupBy("prediction").count().show();

    // save model
//    kmeansModel.write.overwrite().save("/Users/anlor/IdeaProjects/369final/nba_model")

    // load model
//    val kmeansModelLoaded = kmeansModel.load("/Users/anlor/IdeaProjects/369final/nba_model")


    val new_df = NBADF.drop("GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_NAME", "COMMENT", "FGM", "FGA", "FG3M", "FG3A",
      "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "STL", "BLK", "TO", "PF", "PLUS_MINUS")
      .filter("START_POSITION is null and REB is not null and AST is not null and PTS is not null and MIN is not null");


    val temp = new_df.dropDuplicates("PLAYER_ID").limit(200);

    temp.show(50);
    val cols_test = Array("REB", "AST", "PTS");
    val assembler_test = new VectorAssembler().setInputCols(cols_test).setOutputCol("features")
    val featureDf_test = assembler_test.transform(temp)

    featureDf_test.show();

    // prediction with this data of non-starters
    val df3 = kmeansModel.transform(featureDf_test);
    df3.show(50)


  }
}
