package Facilitators
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

// This program takes the top 10 players (in terms of PPG) from the Spurs'(NBA) 2020-2021 roster and predicts if the
// players are 'Primary Facilitators', 'Secondary Facilitators', or 'Tertiary Facilitators' when compared to each other
object MainMethod {
  def main(args: Array[String]): Unit = {
    // turn off the text that appears when running this code
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // read input file
    val conf = new SparkConf().setAppName("369project").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val NBA = sc.textFile("NBA.csv")

    // get rid of headers
    val header1 = NBA.first()
    val nba = NBA.filter(x => x != header1)
    val header2 = nba.first()
    val NEWNBA = nba.filter(x => x != header2)

    // read in the necessary data from the input file into an rdd (only the top 10 scorers are kept for this rdd)
    val linesMap = sc.parallelize(NEWNBA.map(line =>
      (line.split(",")(0).toInt, // Player Id
      (line.split(",")(1),       // Player Name
        line.split(",")(3),      // Player Position
        line.split(",")(8),      // Player USG%
        line.split(",")(21),     // Player APG
        line.split(",")(18)),    // Player PPG
      line.split(",")(2).equals("San"))).filter(_._3.equals(true)) // filter out any team that's not the Spurs
      .map(line => (line._1.toInt, line._2)).sortBy(-_._2._5.toDouble) // sort by players with most points (descending)
    .take(10) // take top 10 players only
    .map(line => (line._1.toInt, (line._2._1, line._2._2, line._2._3, line._2._4)))) // get rid of PPG section in rdd

    linesMap.foreach(println)
    /* linesMap will look like this:
        (116,(DeMar DeRozan,G-F,23.9,7.3))
        (337,(Dejounte Murray,G,23.7,5.2))
        (240,(Keldon Johnson,F-G,20.6,2))
        (5,(LaMarcus Aldridge,C-F,22.7,1.7))
        (325,(Patty Mills,G,20.5,2.5))
        (489,(Derrick White,G,23.1,3.5))
        (477,(Lonnie Walker IV,G-F,19.4,1.7))
        (161,(Rudy Gay,F-G,23.3,1.5))
        (382,(Jakob Poeltl,C,12.3,1.9))
        (471,(Devin Vassell,G-F,13.9,1.1))
    */

    // find assists Leaders' number of assists per game
    var assistsLeader = linesMap.first()._2._4.toDouble
    for (z <- 0 until linesMap.count().toInt) {
      if (linesMap.zipWithIndex().filter(_._2 == z).map(_._1).first()._2._4.toDouble > assistsLeader) {
        assistsLeader = linesMap.zipWithIndex().filter(_._2 == z).map(_._1).first()._2._4.toDouble
      }
    }

    // create a table of stats which holds players' USG% and APG like so; [USG%, APG]
    val stat = linesMap.map(x => Vectors.dense(x._2._3.toDouble, x._2._4.toDouble)).cache()

    // create a k-means model that will run our rdd
    val model = new KMeans()
      .setK(3)
      .setSeed(1000)
      .run(stat)

    // Use helper functions in 'HelperFunctions' file to print the cluster coordinates
    HelperFunctions.printClusters(model, assistsLeader)
    /* The output will look like this:
        Tertiary Facilitators Cluster Coordinate = [13.100000000000001,1.5]
        Secondary Facilitators Cluster Coordinate = [21.6,2.1499999999999995]
        Primary Facilitators Cluster Coordinate = [23.799999999999997,6.25]
    */

    // Use helper functions in 'HelperFunctions' file to classify clusters into either 'Primary Facilitators',
    // 'Secondary Facilitators', & 'Tertiary Facilitators' and then print out our models' predictions for each player
    HelperFunctions.printFacilitators(model, linesMap, assistsLeader, stat)
    /* The output will look like this:
        Primary Facilitators:
        ---------------------
        DeMar DeRozan
        Dejounte Murray

        Secondary Facilitators:
        -----------------------
        Keldon Johnson
        LaMarcus Aldridge
        Patty Mills
        Derrick White
        Lonnie Walker IV
        Rudy Gay

        Tertiary Facilitators:
        ----------------------
        Jakob Poeltl
        Devin Vassell
    */
  }
}
