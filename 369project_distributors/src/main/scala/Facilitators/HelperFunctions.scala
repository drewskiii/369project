package Facilitators
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.rdd.RDD

object HelperFunctions {
  // this function uses our kmeans model and our assists Leaders' assist total to classify clusters and assigns players
  // from our RDD into these cluster categories (assist leader will be in Primary Facilitators)
  def printFacilitators(model : KMeansModel, linesMap : RDD[(Int, (String, String, String, String))],
                        assistsLeader : Double, stat : RDD[org.apache.spark.mllib.linalg.Vector]) : Unit = {
    println("\nPrimary Facilitators:\n---------------------")
    for (y <- 0 until model.predict(stat).count().toInt) {
      if (model.predict(stat).take(model.predict(stat).count().toInt).mkString("", "",
        "")(y).toString.equals(HelperFunctions.whichCoordP(model, assistsLeader.toDouble).toString)) {
        println(linesMap.zipWithIndex().filter(_._2 == y).map(_._1).first()._2._1)
      }
    }
    println("\nSecondary Facilitators:\n-----------------------")
    for (y <- 0 until model.predict(stat).count().toInt) {
      if (model.predict(stat).take(model.predict(stat).count().toInt).mkString("", "",
        "")(y).toString.equals(HelperFunctions.whichCoordS(model, assistsLeader.toDouble).toString)) {
        println(linesMap.zipWithIndex().filter(_._2 == y).map(_._1).first()._2._1)
      }
    }
    println("\nTertiary Facilitators:\n----------------------")
    for (y <- 0 until model.predict(stat).count().toInt) {
      if (model.predict(stat).take(model.predict(stat).count().toInt).mkString("", "",
        "")(y).toString.equals(HelperFunctions.whichCoordT(model, assistsLeader.toDouble).toString)) {
        println(linesMap.zipWithIndex().filter(_._2 == y).map(_._1).first()._2._1)
      }
    }
  }

  // This function maps either a 0, 1, or 2 to our Tertiary Cluster
  def whichCoordT(model: KMeansModel, aL: Double): Int = {
    val mc0 = model.clusterCenters(0).toArray(1)
    val mc1 = model.clusterCenters(1).toArray(1)
    val mc2 = model.clusterCenters(2).toArray(1)

    for (x <- 0 to model.clusterCenters.length) {
      if (x == 2) {
        if ((mc2 - aL).abs < (mc0 - aL).abs && (mc2 - aL).abs < (mc1 - aL).abs) {}
        else if ((mc2 - aL).abs < (mc0 - aL).abs && (mc2 - aL).abs > (mc1 - aL).abs ||
          (mc2 - aL).abs > (mc0 - aL).abs && (mc2 - aL).abs < (mc1 - aL).abs) {}
        else {
          return 2
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 0) {
        if ((mc0 - aL).abs < (mc2 - aL).abs && (mc0 - aL).abs < (mc1 - aL).abs) {}
        else if ((mc0 - aL).abs < (mc1 - aL).abs && (mc0 - aL).abs > (mc2 - aL).abs ||
          (mc0 - aL).abs > (mc1 - aL).abs && (mc0 - aL).abs < (mc2 - aL).abs) {}
        else {
          return 0
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 1) {
        if ((mc1 - aL).abs < (mc2 - aL).abs && (mc1 - aL).abs < (mc0 - aL).abs) {}
        else if ((mc1 - aL).abs < (mc0 - aL).abs && (mc1 - aL).abs > (mc2 - aL).abs ||
          (mc1 - aL).abs > (mc0 - aL).abs && (mc1 - aL).abs < (mc2 - aL).abs) {}
        else {
          return 1
        }
      }
    }
    return 0
  }

  // This function maps either a 0, 1, or 2 to our Secondary Cluster
  def whichCoordS(model: KMeansModel, aL: Double): Int = {
    val mc0 = model.clusterCenters(0).toArray(1)
    val mc1 = model.clusterCenters(1).toArray(1)
    val mc2 = model.clusterCenters(2).toArray(1)

    for (x <- 0 to model.clusterCenters.length) {
      if (x == 2) {
        if ((mc2 - aL).abs < (mc0 - aL).abs && (mc2 - aL).abs > (mc1 - aL).abs ||
          (mc2 - aL).abs > (mc0 - aL).abs && (mc2 - aL).abs < (mc1 - aL).abs) {
          return 2
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 0) {
        if ((mc0 - aL).abs < (mc1 - aL).abs && (mc0 - aL).abs > (mc2 - aL).abs ||
          (mc0 - aL).abs > (mc1 - aL).abs && (mc0 - aL).abs < (mc2 - aL).abs) {
          return 0
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 1) {
        if ((mc1 - aL).abs < (mc0 - aL).abs && (mc1 - aL).abs > (mc2 - aL).abs ||
          (mc1 - aL).abs > (mc0 - aL).abs && (mc1 - aL).abs < (mc2 - aL).abs) {
          return 1
        }
      }
    }
    return 0
  }

  // This function maps either a 0, 1, or 2 to our Primary Cluster
  def whichCoordP(model: KMeansModel, aL: Double): Int = {
    val mc0 = model.clusterCenters(0).toArray(1)
    val mc1 = model.clusterCenters(1).toArray(1)
    val mc2 = model.clusterCenters(2).toArray(1)

    for (x <- 0 to model.clusterCenters.length) {
      if (x == 2) {
        if ((mc2 - aL).abs < (mc0 - aL).abs && (mc2 - aL).abs < (mc1 - aL).abs) {
          return 2
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 0) {
        if ((mc0 - aL).abs < (mc2 - aL).abs && (mc0 - aL).abs < (mc1 - aL).abs) {
          return 0
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 1) {
        if ((mc1 - aL).abs < (mc2 - aL).abs && (mc1 - aL).abs < (mc0 - aL).abs) {
          return 1
        }
      }
    }
    return 0
  }

  // this function prints out the coordinates for all three of our clusters (it uses the same logic as some of the
  // functions defined above to determine which clusters are 'Primary', 'Secondary', and 'Tertiary')
  def printClusters(model: KMeansModel, aL: Double): Unit = {
    val mc0 = model.clusterCenters(0).toArray(1)
    val mc1 = model.clusterCenters(1).toArray(1)
    val mc2 = model.clusterCenters(2).toArray(1)

    for (x <- 0 to model.clusterCenters.length) {
      if (x == 2) {
        if ((mc2 - aL).abs < (mc0 - aL).abs && (mc2 - aL).abs < (mc1 - aL).abs) {
          print("Primary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
        else if ((mc2 - aL).abs < (mc0 - aL).abs && (mc2 - aL).abs > (mc1 - aL).abs ||
          (mc2 - aL).abs > (mc0 - aL).abs && (mc2 - aL).abs < (mc1 - aL).abs) {
          print("Secondary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
        else {
          print("Tertiary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 0) {
        if ((mc0 - aL).abs < (mc2 - aL).abs && (mc0 - aL).abs < (mc1 - aL).abs) {
          print("Primary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
        else if ((mc0 - aL).abs < (mc1 - aL).abs && (mc0 - aL).abs > (mc2 - aL).abs ||
          (mc0 - aL).abs > (mc1 - aL).abs && (mc0 - aL).abs < (mc2 - aL).abs) {
          print("Secondary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
        else {
          print("Tertiary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
      }
    }
    for (x <- 0 to model.clusterCenters.length) {
      if (x == 1) {
        if ((mc1 - aL).abs < (mc2 - aL).abs && (mc1 - aL).abs < (mc0 - aL).abs) {
          print("Primary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
        else if ((mc1 - aL).abs < (mc0 - aL).abs && (mc1 - aL).abs > (mc2 - aL).abs ||
          (mc1 - aL).abs > (mc0 - aL).abs && (mc1 - aL).abs < (mc2 - aL).abs) {
          print("Secondary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
        else {
          print("Tertiary Facilitators Cluster Coordinate = ")
          println(model.clusterCenters(x))
        }
      }
    }
  }

}
