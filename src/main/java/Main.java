import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;

import static org.apache.spark.sql.functions.udf;


public class Main {

    private static JavaSparkContext getSparkContext(boolean onServer) {
        SparkConf sparkConf = new SparkConf().setAppName("2AMD15");
        if (!onServer) sparkConf = sparkConf.setMaster("local[*]");
        JavaSparkContext javaSparkContext = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(sparkConf));

        // TODO: You may want to change ERROR to WARN to receive more info. For larger data sets, to not set the
        // log level to anything below WARN, Spark will print too much information.
        javaSparkContext.setLogLevel("ERROR");

        return javaSparkContext;
    }

    private static Dataset q1a(JavaSparkContext sparkContext, boolean onServer) {
        String vectorsFilePath = (onServer) ? "/vectors.csv" : "vectors.csv";

        SparkSession sparkSession = SparkSession.builder().sparkContext(sparkContext.sc()).getOrCreate();

        return sparkSession.read().csv(vectorsFilePath);
    }

    private static JavaRDD q1b(JavaSparkContext sparkContext, boolean onServer) {
        String vectorsFilePath = (onServer) ? "/vectors.csv" : "vectors.csv";

        SparkSession sparkSession = SparkSession.builder().sparkContext(sparkContext.sc()).getOrCreate();

        return sparkSession.sparkContext()
        .textFile(vectorsFilePath, 1)
        .toJavaRDD();
    }

    private static void q2(JavaSparkContext sparkContext, Dataset dataset) {
        dataset.createOrReplaceTempView("dataset");

        Dataset<Row> splitted = dataset.sqlContext().sql("SELECT _c0 as id, SPLIT(_c1, ';') AS values FROM dataset");
        splitted.createOrReplaceTempView("splitted_data");

        System.out.println("Number of vectors: " + splitted.count());
        splitted.show(10);

        UserDefinedFunction sum_var = udf(
                (Seq<String> a, Seq<String> b, Seq<String> c) -> {
                    // Convert Seq to List
                    List<String> aa = JavaConverters.asJava(a);
                    List<String> bb = JavaConverters.asJava(b);
                    List<String> cc = JavaConverters.asJava(c);

                    // Aggregate the vectors
                    int[] result = new int[a.length()];
                    for (int i = 0; i < a.length(); i++) {
                        result[i] = Integer.parseInt(aa.get(i)) + Integer.parseInt(bb.get(i)) + Integer.parseInt(cc.get(i));
                    }

                    // Compute the variance
                    OptionalDouble optAvg = Arrays.stream(result).average();
                    if (optAvg.isEmpty()) {
                        return 0;
                    }
                    double avg = optAvg.getAsDouble();
                    double var = 0;
                    for (int j : result) {
                        var += Math.pow(j, 2);
                    }
                    var = var / result.length;
                    return var - Math.pow(avg, 2);
                }, DataTypes.DoubleType
        );

        // Register the function as a UDF
        dataset.sqlContext().udf().register("sum_var", sum_var);

        // Join the dataset with itself twice to get all possible triplets
        Dataset<Row> triplets = dataset.sqlContext().sql(
                "SELECT X.id as X_id, Y.id as Y_id, Z.id as Z_id, " +
                "sum_var(X.values, Y.values, Z.values) AS var " +
                "FROM " +
                "(splitted_data as X JOIN splitted_data AS Y ON X.id < Y.id JOIN splitted_data AS Z ON Y.id < Z.id)" +
                "ORDER BY var ASC");
        triplets.createOrReplaceTempView("triplets");
        // Persist the triplets so that we can reuse them for different values of tau
        triplets.persist();

        // Print the size of the triplets
        System.out.println("Number of triplets: " + triplets.count());
        triplets.show(10);

        int[] taus = { 20, 50, 310, 360, 410 };
        for (int tau : taus) {
            System.out.println("Tau: " + tau);
            // Compute the variance of each vector
            Dataset<Row> variance = dataset.sqlContext().sql("SELECT " +
                    "X_id, Y_id, Z_id, var " +
                    "FROM " +
                    "triplets " +
                    "WHERE var <= " + tau);
            variance.createOrReplaceTempView("variance");

            // Print the size of the variance
            System.out.println("Number of triplets: " + variance.count());
            variance.show(10);
        }

        // Unpersist the triplets
        triplets.unpersist();
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
