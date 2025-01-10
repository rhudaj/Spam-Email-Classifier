package spamclassifier
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._
import org.apache.hadoop.conf.Configuration
import scala.math.exp
import scala.collection.mutable
import scala.util.Random

// PMI configuration for both stripes & pairs approaches.
class TrainConf(args: Seq[String]) extends ScallopConf(args) {
  	mainOptions = Seq(input, model, shuffle)
  	val input = opt[String](descr = "input train data", required = true)
  	val model = opt[String](descr = "model output dir", required = true)
	val shuffle = toggle("shuffle")
  	verify()
}

object Train {
  	val log = Logger.getLogger(getClass().getName())

	// Train the model
	def train(sc: SparkContext, inputPath: String, outputPath: String, shuffle: Boolean) {

		// Model parameters (weights)
		val w = mutable.Map[Int, Double]()

		// step/learn rate
		val delta = 0.002

		// Helper: Computes the spam score, for a SINGLE feature
		def spamminess(features: Array[Int]) : Double = {
			var score = 0d
			features.foreach(f => if (w.contains(f)) score += w(f))
			score
		}

		val numPartitions = 1 	// prevent re-ordering of input data
		var trainData = sc.textFile(inputPath, numPartitions)

		// Shuffle the data in random order (if specified)
		if (shuffle) {
			trainData = trainData
			// assign each line a random key
			.map(line =>{
				(Random.nextDouble(), line)
			})
			// Sort by the random order
			.sortByKey()
			// Dont need the key anymore
			.map(line => line._2)
		}

		// Train the model

		// Extract (label, feature) from each line of text
		trainData.map( line => {
			val arr = line.split(" ")
			val isSpam = if (arr(1).equals("spam")) 1 else 0
			val features = arr.drop(2).map(e => e.toInt)
			// All same key (0) => 1 reducer
			(0, (isSpam, features) )
		})
		// All data to a single reducer
		.groupByKey(1)
		// Train the model on the single reducer
		.flatMap(kvp => {
			kvp._2.foreach( v => {
				val isSpam = v._1
				val features = v._2
				// Get the prediction
				val score = spamminess(features)
				// Use the logistic function
				val prob = 1.0 / (1 + Math.exp(-score))
				// Update the weighting for each feature
				features.foreach(f => {
					w(f) = w.getOrElse(f, 0.0) + (isSpam - prob) * delta
				})
			})
			w
		})
		// OUTPUT RESULTS
		.saveAsTextFile(outputPath)
	}

  	def main(argv: Array[String]) {

		// CMD LINE ARGS

			val args = new TrainConf(argv)

			log.info("Input: " + args.input())
			log.info("Out path: " + args.model())
			log.info("Shuffle: " + args.shuffle.isSupplied)

		// SETUP Spark

			// SparkContext represents the connection to a Spark cluster
			val sc = new SparkContext(
				new SparkConf().setAppName("Train")
			)

		// Delete Output directory (if already exists)

			FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.model()), true)

		// Train the Model

			train(sc, args.input(), args.model(), args.shuffle.isSupplied)
	}
}