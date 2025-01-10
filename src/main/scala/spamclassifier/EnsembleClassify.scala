package spamclassifier
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.rogach.scallop._
import scala.collection.mutable.Map

import org.apache.log4j.Logger
import org.apache.log4j.Level

class EnsembleConf(args: Seq[String]) extends ScallopConf(args) {
    mainOptions = Seq(input, output, model)
    val input = opt[String](descr = "input path", required = true)
    val output = opt[String](descr = "output path", required = true)
    val model = opt[String](descr = "model path", required = true)
    val method = opt[String](descr = "ensemple aggregate method", required = true)
    verify()
}

object EnsembleClassify {
    val log = Logger.getLogger(getClass().getName())
    def getModel(sc: SparkContext, modelPath: String): scala.collection.Map[Int,Double] = {
        // get the model from a file, return the KVPs in a Map
        sc.textFile(modelPath)
        .map(line => {
            val feat_weight = line.substring(1, line.length-1).split(",")
            (feat_weight(0).toInt, feat_weight(1).toDouble)
        })
        // Return the key-value pairs in this RDD to the master as a Map.
        .collectAsMap
    }

    def main(argv: Array[String]) {

        // SETUP STUFF

            val args = new EnsembleConf(argv)

            log.info("Input path: " + args.input())
            log.info("Output path: " + args.output())
            log.info("Model path: " + args.model())
            log.info("Method of ensemble: " + args.method())

            val conf = new SparkConf().setAppName("EnsembleClassify")
            val sc = new SparkContext(conf)

            val testData = sc.textFile(args.input())
            val outputDir = new Path(args.output())

            FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

            val modelsPath = args.model()
            val ensemble = sc.broadcast(Array(
                getModel(sc, modelsPath+"/part-00000"),
                getModel(sc, modelsPath+"/part-00001"),
                getModel(sc, modelsPath+"/part-00002")
            ))
            val method = sc.broadcast(args.method())

        // Make predictions

        testData
        // Parse the test data
        .map(line =>{
            val arr = line.split(" ")
            val features = arr.drop(2).map(e => e.toInt)
            // Get total score by model
            val scores = Array(0.0, 0.0, 0.0)
            features.foreach(f => {
                var i = 0
                ensemble.value.foreach(m => {
                    if(m.contains(f)) scores(i) += m(f.toInt)
                    i+=1
                })
            })
            // prediction depends on chosen method
            if(method.value.equals("average")) {
                val avg_score = scores.sum / scores.length
                val pred = if ( avg_score > 0) "spam" else "ham"
                (arr(0), arr(1), avg_score, pred)
            } else {
                val spamVotes = scores.count(_ > 0)
                val pred = if (2*spamVotes > scores.length) "spam" else "ham"
                (arr(0), arr(1), spamVotes, pred)
            }
        })
        .saveAsTextFile(args.output())
    }
}