import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import java.util.Arrays;


public class Main {

    private static JavaSparkContext getSparkContext(boolean onServer) {
        SparkConf sparkConf = new SparkConf().setAppName("2AMD15");
        if (!onServer) sparkConf = sparkConf.setMaster("local[*]");
        JavaSparkContext javaSparkContext = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(sparkConf));

        // TODO: You may want to change ERROR to WARN to receive more info. For larger data sets, to not set the
        // log level to anything below WARN, Spark will print too much information.
        if (onServer) javaSparkContext.setLogLevel("ERROR");

        return javaSparkContext;
    }

    private static Dataset q1a(JavaSparkContext sparkContext, boolean onServer) {
        String vectorsFilePath = (onServer) ? "/vectors.csv" : "vectors.csv";

        SparkSession sparkSession = SparkSession.builder().sparkContext(sparkContext.sc()).getOrCreate();

        Dataset<Row> df = sparkSession.read().csv(vectorsFilePath);

        df.show();

        return df;
    }

    private static JavaRDD q1b(JavaSparkContext sparkContext, boolean onServer) {
        String vectorsFilePath = (onServer) ? "/vectors.csv" : "vectors.csv";

        SparkSession sparkSession = SparkSession.builder().sparkContext(sparkContext.sc()).getOrCreate();

        JavaRDD<String> RDD = sparkSession.sparkContext()
        .textFile(vectorsFilePath, 1)
        .toJavaRDD();
        
        JavaRDD<String> words = RDD.flatMap(line -> Arrays.asList(line.split(";")).iterator());
        long count = words.count();
        System.out.println("Word count: " + count);

        return RDD;

    }

    private static void q2(JavaSparkContext sparkContext, Dataset dataset) {
        //Dataset<Row> first_column = dataset.select("_c0");
        //Dataset<Row> second_column = dataset.select("_c1");

        dataset.createOrReplaceTempView("dataset");

        Dataset<Row> sql = dataset.sqlContext().sql("SELECT _c0 from dataset");
        sql.show();

        // Split the values field into an array of integers
        dataset.sqlContext().sql("SELECT _c0 as id, SPLIT(_c1, ';') AS values FROM dataset")
            .createOrReplaceTempView("splitted_data");
                
        dataset.sqlContext().sql("SELECT X.id,Y.id,Z.id AS _values FROM splitted_data as X " + 
        "JOIN splitted_data AS Y ON X.id < Y.id JOIN splitted_data AS Z ON Y.id < Z.id")
        .show();
        // Compute the variance of each vector
    //     dataset.sqlContext().sql("SELECT " +
    //     "triplets.X_numbers, " +
    //     "triplets.Y_numbers, " +
    //     "triplets.Z_numbers, " +
    //     "variance(cast(pow(cast(triplets.X_numbers[0] as double), 2) as double) + " +
    //     "         cast(pow(cast(triplets.Y_numbers[0] as double), 2) as double) + " +
    //     "         cast(pow(cast(triplets.Z_numbers[0] as double), 2) as double)) as varianceXYZ " +
    //   "FROM ( " +
    //     "SELECT " +
    //       "t1.ID as X, " +
    //       "t2.ID as Y, " +
    //       "t3.ID as Z, " +
    //       "t1.numbers as X_numbers, " +
    //       "t2.numbers as Y_numbers, " +
    //       "t3.numbers as Z_numbers " +
    //     "FROM " +
    //       "( " +
    //         "SELECT " +
    //           "_c0 as ID, " +
    //           "split(_c1, ';') as numbers " +
    //         "FROM " +
    //           "dataset " +
    //       ") t1 " +
    //     "JOIN " +
    //       "( " +
    //         "SELECT " +
    //           "_c0 as ID, " +
    //           "split(_c1, ';') as numbers " +
    //         "FROM " +
    //           "dataset " +
    //       ") t2 ON t1.ID < t2.ID " +
    //     "JOIN " +
    //       "( " +
    //         "SELECT " +
    //           "_c0 as ID, " +
    //           "split(_c1, ';') as numbers " +
    //         "FROM " +
    //           "dataset " +
    //       ") t3 ON t2.ID < t3.ID " +
    //   ") AS triplets " +
    //   "GROUP BY triplets.X_numbers, triplets.Y, triplets.Z " +
    //   "HAVING varianceXYZ <= 410 ")
    //   .show();

    }

    private static void q3(JavaSparkContext sparkContext, JavaRDD rdd) {
        // TODO: Implement Q3 here
    }

    private static void q4(JavaSparkContext sparkContext, JavaRDD rdd) {
        // TODO: Implement Q4 here
    }


    // Main method which initializes a Spark context and runs the code for each question.
    // To skip executing a question while developing a solution, simply comment out the corresponding method call.
    public static void main(String[] args) {

        boolean onServer = false; // TODO: Set this to true if and only if building a JAR to run on the server

        JavaSparkContext sparkContext = getSparkContext(onServer);

        Dataset dataset = q1a(sparkContext, onServer);

        JavaRDD rdd = q1b(sparkContext, onServer);

        q2(sparkContext, dataset);

        q3(sparkContext, rdd);

        q4(sparkContext, rdd);

        sparkContext.close();


    }
}
