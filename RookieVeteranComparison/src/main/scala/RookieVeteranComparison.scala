import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object RookieVeteranComparison {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("Name").setMaster("local[1]")
    val sc = new SparkContext(conf)
    // name, year_start
    val playerStats = sc.textFile("player_data1.csv").map(x => x.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1).toList)
    var header = playerStats.first()
    val players = playerStats.filter(x => x != header).map(x => (x(0), x(1).toInt)).distinct()
    val rookies = players.filter(x => x._2 >= 2016)
    val veterans = players.filter(x => x._2 < 2016)

    // player, TS%, ORB%,DRB%,TRB%,AST%,STL%,BLK%,TOV%,USG%,FG%
    // 2     , 10 , 13  ,14  ,15  ,16  ,17  ,18  ,19  ,20  ,33
    val seasonStatsInput = sc.textFile("Seasons_Stats.csv").map(x => x.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1).toList)
    header = seasonStatsInput.first()
    val veteranStats = seasonStatsInput.filter(x => x != header)
      .filter(x => x(1) == "2017")
      //      .map(x => List(x(2), x(10), x(13), x(14), x(15), x(16), x(17), x(18), x(19), x(20), x(33)))
      .map(x => List(x(2), x(16), x(20)))
      .map(x => x.map(y => if (y == "") "0" else y))
      .map(x => (x.head, x.tail))
      .join(veterans).map(x => x._2._1)
      .map(x => Vectors.dense(x.map(_.toDouble).toArray)).cache()
    val rookieStats = seasonStatsInput.filter(x => x != header)
      .filter(x => x(1) == "2017")
//      .map(x => List(x(2), x(10), x(13), x(14), x(15), x(16), x(17), x(18), x(19), x(20), x(33)))
      .map(x => List(x(2), x(16), x(20)))
      .map(x => x.map(y => if (y == "") "0" else y))
      .map(x => (x.head, x.tail))
      .join(rookies).map(x => (x._1, x._2._1, x._2._2))
      .map(x => (x._1, Vectors.dense(x._2.map(_.toDouble).toArray), x._3)).cache()
//    rookieStats.foreach(println)

    val model = new KMeans()
      .setK(5)
      .setSeed(100)
      .run(veteranStats)

    model.clusterCenters.foreach(println)
    println("Veteran")
    val veteranTotal = veteranStats.collect().size
    model.predict(veteranStats).map(x => (x, 0)).countByKey()
      .map(x => (x._1, x._2*1.0/veteranTotal, x._2)).foreach(println)
//    rookieStats.foreach(x => println(x._1, model.predict(x._2), x._3))
    println("Rookie")
    val rookieTotal = rookieStats.collect().size
    model.predict(rookieStats.map(x => x._2)).map(x => (x, 0)).countByKey()
      .map(x => (x._1, x._2*1.0/rookieTotal, x._2)).foreach(println)
  }
}
