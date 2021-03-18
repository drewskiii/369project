import org.apache.spark.SparkContext._

import scala.io._
import org.apache.spark.rdd._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

import scala.collection._


object App {
  def main(args: Array[String]):Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    storeinfo()
  }

  def storeinfo(): Unit= {
    val conf = new SparkConf().setAppName("name").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val playerHeights = sc.textFile("all_seasons.csv").map(line => (line.split(",")(1), line.split(",")(4).toDouble))
      .sortBy(r => r._2) //(name, height)
    val size = playerHeights.collect().length/3
    val data = 0 :: List.fill(size)(0) ::: List.fill(size)(1) ::: List.fill(size)(2)
    //splitting the heights into three categorical groups

    val games = sc.textFile("games_details.csv").map(line => (line.split(",")(5),
      (line.split(",")(10).toDouble, line.split(",")(13).toDouble, line.split(",")(16).toDouble, line.split(",")(25).toDouble)))
    //(name, (3pt percentage, free throw percentage, rebounds, num pts))

    val aggregatedData = games.groupByKey().map({case (k,v) => (k, averages(v.toList))}) //(name, (avg 3pt prct, avg free throw prct, avg rebounds, avg pts))

    val testdata = playerHeights.join(aggregatedData).map({case (name, (height, (three, free, rebound, pts))) => List(three,free,rebound, pts)})
      .map(r => Vectors.dense(r.toArray)).cache()

    val numClusters = 3
    val numIterations = 20

    val clusters = KMeans.train(testdata, numClusters, numIterations)

    //predicts if grouped correctly
    val values = clusters.predict(testdata).collect()
    values.foreach(v => print(v + " "))
    var fp = 0
    var fn = 0
    var tp = 0
    for(i <- values.indices) {
      for (j <- values.indices) {
        if(values.toList(i) == values.toList(j) && data(i) == data(j)) {
          tp = tp + 1
        } else if (values.toList(i) == values.toList(j) && data(i) != data(j)) {
          fp = fp + 1
        } else if (values.toList(i) != values.toList(j) && data(i) == data(j)) {
          fn = fn + 1
        }
      }
    }
    val p = tp/(tp+fp)
    val r =  tp/(tp+fn)
    val f1 = (2*p*r)/(p+r)
    println("Precision: " + p)
    println("Recall: "+r)
    println("f1: " + f1)
  }
  def averages(elements: List[(Double, Double, Double, Double)]): (Double, Double, Double, Double)= {
    val ans = elements.fold(0.0,0.0,0.0,0.0) ((total, n) => (total._1 + n._1, total._2+ n._2, total._3 + n._3, total._4 + n._4))
    (ans._1/elements.size, ans._2/elements.size, ans._3/elements.size, ans._4/elements.size)
  }
}
