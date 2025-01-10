package spamclassifier
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.rogach.scallop._
import org.apache.log4j.Logger
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext

class ClassifyConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output, model)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val model = opt[String](descr = "model path", required = true)
  verify()
}

object Classify {

	val log = Logger.getLogger(getClass().getName())

	def classify(sc: SparkContext, inPath: String, modelPath: String, outPath: String) {

		val model = sc.textFile(modelPath)
			// Parse the model lines to a Map
			.map(line => {
				val feat_weight = line.substring(1, line.length-1).split(",")
				(feat_weight(0).toInt, feat_weight(1).toDouble)
			})
			// Return the key-value pairs in this RDD to the master as a Map.
			.collectAsMap

		// Broadcast the model to Spark cluster (read-only)
		val bcModel = sc.broadcast(model)

		// Make predictions for the test data
		sc.textFile(inPath)
			.map(line =>{
				val arr = line.split(" ")
				val features = arr.drop(2).map(e => e.toInt)
				// Accumulate the total score:
				var totalScore = 0d
				features.foreach(f => {
					if (bcModel.value.contains(f))
						totalScore += bcModel.value(f.toInt)
				})
				// Use threshold for prediction (logstic not necessary)
				val predict = if (totalScore > 0) "spam" else "ham"
				// docID, trueLable, score, predictLabel
				(arr(0), arr(1), totalScore, predict)
			})
			// Output results
			.saveAsTextFile(outPath)
	}

    def main(argv: Array[String]) {

		// Cmd Line Arguments

      	val args = new ClassifyConf(argv)

      	log.info("Input: " + args.input())
      	log.info("Output: " + args.output())
      	log.info("Model path: " + args.model())

		// Setup Spark

		val sc = new SparkContext(
			new SparkConf().setAppName("Classify")
		)

		// Delete Output directory (if already exists)

    	FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)

		// Classify Using the model

		classify(sc, args.input(), args.model()+"/part-00000", args.output())
  	}
}