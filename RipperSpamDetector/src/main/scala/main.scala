import java.io.File
import java.nio.charset.CodingErrorAction
import java.text.DecimalFormat
import java.util
import java.util.Random

import weka.classifiers.evaluation.Evaluation
import weka.classifiers.rules.JRip
import weka.core.{Attribute, Instance, Instances}
import weka.core.converters.CSVLoader
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.io.Codec
import collection.JavaConverters._
import weka.filters.unsupervised.attribute.NumericToNominal

object main {

  def readData(filename:String):Instances={
    val csv_loader = new CSVLoader()
    csv_loader.setSource(new File(filename))
    val data_set_csv = csv_loader.getDataSet()

    val remove_filter = new Remove()
    remove_filter.setAttributeIndicesArray(Array(2,3))
    remove_filter.setInvertSelection(true)
    remove_filter.setInputFormat(data_set_csv)
    val filtered_data = Filter.useFilter(data_set_csv, remove_filter)
    filtered_data.setClassIndex(0)

    val convert = new NumericToNominal()
    convert.setAttributeIndicesArray(Array(0))
    convert.setInputFormat(filtered_data)
    val data_set = Filter.useFilter(filtered_data, convert)
    data_set
  }

  def main(args: Array[String]): Unit = {
    val data_set = readData("../Data/spam_or_not_spam_cleaned.csv")
    // check if data is correctly loaded
    println(data_set.firstInstance())

    val jRip = new JRip()
    jRip.setUsePruning(true)
    jRip.setDebug(true)
    jRip.buildClassifier(data_set)

    val eval = new Evaluation(data_set)
    eval.crossValidateModel(jRip, data_set, 10, new Random())
    val padding = "%1$d\t%2$-10.2f\t%3$-5.2f\t%4$1.2f"

    println(" \tPrecision\tRecall\tF")
    println(padding.format(0, eval.precision(0),eval.recall(0),eval.fMeasure(0)))
    println(padding.format(1, eval.precision(1),eval.recall(1),eval.fMeasure(1)))

    val conf_matrix = eval.confusionMatrix()
    val total = conf_matrix.map(u => u.sum).sum
    val succ = conf_matrix(0)(0) + conf_matrix(1)(1)
    println("Accuracy: "+  (succ/total))

  }

}
