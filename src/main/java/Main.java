import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.collection.Seq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import scala.Tuple2;
import org.apache.spark.util.sketch.CountMinSketch;


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

        return sparkContext.textFile(vectorsFilePath);
    }

    private static void q2(JavaSparkContext sparkContext, Dataset dataset) {
        int[] taus = { 20, 50, 310, 360, 410 };

        dataset.createOrReplaceTempView("dataset");

        Dataset<Row> splitted = dataset.sqlContext().sql("SELECT _c0 as id, SPLIT(_c1, ';') AS values FROM dataset");
        splitted.createOrReplaceTempView("splitted_data");

        UserDefinedFunction sum_var = udf(
                (Seq<String> a, Seq<String> b, Seq<String> c) -> {

                    int[] result = IntStream.range(0, a.size())
                            .map(i -> Integer.parseInt(a.apply(i)) + Integer.parseInt(b.apply(i)) + Integer.parseInt(c.apply(i)))
                            .toArray();

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
                "SELECT X_id, Y_id, Z_id, var FROM (" +
                "SELECT X.id as X_id, Y.id as Y_id, Z.id as Z_id, " +
                "sum_var(X.values, Y.values, Z.values) AS var " +
                "FROM " +
                "(splitted_data as X JOIN splitted_data AS Y ON X.id < Y.id JOIN splitted_data AS Z ON Y.id < Z.id)" +
                ") WHERE var <= " + Arrays.stream(taus).max().getAsInt());
        triplets.createOrReplaceTempView("triplets");
        // Persist the triplets so that we can reuse them for different values of tau
        triplets.persist();

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

        triplets.explain();

        // Unpersist the triplets
        triplets.unpersist();
    }

    private static void q3(JavaSparkContext sparkContext, JavaRDD<String> rdd) {
        int[] taus = { 20, 50, 310, 360, 410 };

        // Split every entry into an id and a vector
        JavaPairRDD<String, int[]> vectors = rdd.mapToPair(x -> {
            String[] split = x.split(",");
            int[] vector = Arrays.stream(split[1].split(";")).mapToInt(Integer::parseInt).toArray();
            return new Tuple2<>(split[0], vector);
        });

        // Join the RDD with itself to get all possible pairs, and filter out the pairs where the first id is smaller than the second
        JavaPairRDD<Tuple2<String, int[]>, Tuple2<String, int[]>> pairs =
                vectors.cartesian(vectors).filter(x -> x._1._1.compareTo(x._2._1) < 0);

        // Join it again with itself to get all possible triplets
        JavaPairRDD<Tuple2<Tuple2<String, int[]>, Tuple2<String, int[]>>, Tuple2<String, int[]>> triplets =
                pairs.cartesian(vectors).filter(x -> x._1._2._1.compareTo(x._2._1) < 0);

        // Sum the vectors of each triplet, and create a new id for the triplet, which is the concatenation of the ids of the vectors
        JavaPairRDD<String, int[]> summed = triplets.mapToPair(x -> {
            int[] sum = new int[x._1._1._2.length];
            for (int i = 0; i < sum.length; i++) {
                sum[i] = x._1._1._2[i] + x._1._2._2[i] + x._2._2[i];
            }
            return new Tuple2<>(x._1._1._1 + x._1._2._1 + x._2._1, sum);
        });

        // Compute the variance of each vector
        JavaPairRDD<String, Double> variance = summed.mapValues(x -> {
            // Compute the average
            double avg = Arrays.stream(x).average().getAsDouble();
            // Compute E[X^2]
            double ex2 = Arrays.stream(x).mapToDouble(i -> Math.pow(i, 2)).sum() / ((double) x.length);
            // Compute (E[X])^2
            double exSquared = Math.pow(avg, 2);
            return ex2 - exSquared;
        });

        // Only keep the triplets where the variance is smaller than the biggest tau
        JavaPairRDD<String, Double> filtered = variance.filter(x -> x._2 <= Arrays.stream(taus).max().getAsInt());

        filtered.persist(StorageLevel.MEMORY_ONLY());

        for (int tau : taus) {
            System.out.println("Tau: " + tau);
            // Filter out the triplets where the variance is smaller than tau
            JavaPairRDD<String, Double> tauFiltered = filtered.filter(x -> x._2 <= tau);
            // Print the number of triplets
            System.out.println("Number of triplets: " + tauFiltered.count());
            // Print the first 10 triplets
            tauFiltered.take(10).forEach(System.out::println);
        }

        filtered.unpersist();
    }

    private static void q3_old(JavaSparkContext sparkContext, JavaRDD<String> rdd) {
        int[] taus = { 20, 50, 310, 360, 410 };

        // Split every entry into an id and a vector
        JavaRDD<Tuple2<String, int[]>> splitted = rdd.map(x -> {
            String[] split = x.split(",");
            int[] vector = Arrays.stream(split[1].split(";")).mapToInt(Integer::parseInt).toArray();
            return new Tuple2<>(split[0], vector);
        });

        splitted.persist(StorageLevel.MEMORY_ONLY());

        // Join the RDD with itself to get all possible pairs, and filter out the pairs where the first id is smaller than the second
        JavaPairRDD<Tuple2<String, int[]>, Tuple2<String, int[]>> joined =
                splitted.cartesian(splitted).filter(x -> x._1._1.compareTo(x._2._1) < 0);
        // Join it again with itself to get all possible triplets
        JavaPairRDD<Tuple2<Tuple2<String, int[]>, Tuple2<String, int[]>>, Tuple2<String, int[]>> joined2 =
                joined.cartesian(splitted).filter(x -> x._1._2._1.compareTo(x._2._1) < 0);

        splitted.unpersist();

        // Sum the vectors of each triplet, and create a new id for the triplet, which is the concatenation of the ids of the vectors
        JavaPairRDD<String, int[]> summed = joined2.mapToPair(x -> {
            int[] sum = IntStream.range(0, x._1._1._2.length)
                    .map(i -> x._1._1._2[i] + x._1._2._2[i] + x._2._2[i])
                    .toArray();
            return new Tuple2<>(x._1._1._1 + x._1._2._1 + x._2._1, sum);
        });

        // Compute the variance of each vector
        JavaPairRDD<String, Double> variance = summed.mapValues(x -> {
            // Compute the average
            double avg = Arrays.stream(x).average().getAsDouble();
            // Compute the variance
            double var = 0;
            for (int i : x) {
                var += Math.pow(i, 2);
            }
            var = var / x.length;
            return var - Math.pow(avg, 2);
        });

        // Only keep the triplets where the variance is smaller than the biggest tau
        variance = variance.filter(x -> x._2 <= Arrays.stream(taus).max().getAsInt());
        variance.persist(StorageLevel.MEMORY_ONLY());

        for (int tau : taus) {
            System.out.println("Tau: " + tau);
            // Filter out the triplets where the variance is smaller than tau
            JavaPairRDD<String, Double> filtered = variance.filter(x -> x._2 <= tau);
            // Print the first 10 triplets, formatted nicely
            filtered.take(10).forEach(x -> {
                System.out.println(x._1 + " " + x._2);
            });
            // Print the number of triplets
            System.out.println("Number of triplets: " + filtered.count());
        }

        variance.unpersist();
    }

    private static void q4(JavaSparkContext sparkContext, JavaRDD<String> rdd, int[] taus, double eps, double delta, Boolean lower) {

        // Split every entry into an id and a vector
        JavaRDD<Tuple2<String, int[]>> splitted = rdd.map(x -> {
            String[] split = x.split(",");
            int[] vector = Arrays.stream(split[1].split(";")).mapToInt(Integer::parseInt).toArray();
            return new Tuple2<>(split[0], vector);
        });

        // Initialize the Count-Min sketch and  find a way to use this sketch for representing the splitted RDD 
        CountMinSketch cms = CountMinSketch.create(eps, delta, 1);
        JavaPairRDD<String, CountMinSketch> cmsRDD = splitted.mapToPair(x -> {
            cms.add(x._1, 1);
            return new Tuple2<>(x._1, cms);
        });
        // Use the sketch to get an estimate of the number of distinct ids in the RDD
        // Print the estimate
        System.out.println("Estimated number of distinct ids: " + cmsRDD.count());

        // Join the RDD with itself to get all possible pairs, and filter out the pairs where the first id is smaller than the second
        JavaPairRDD<Tuple2<String, int[]>, Tuple2<String, int[]>> joined =
                splitted.cartesian(splitted).filter(x -> x._1._1.compareTo(x._2._1) < 0);
        // Join it again with itself to get all possible triplets
        JavaPairRDD<Tuple2<Tuple2<String, int[]>, Tuple2<String, int[]>>, Tuple2<String, int[]>> joined2 =
                joined.cartesian(splitted).filter(x -> x._1._2._1.compareTo(x._2._1) < 0);

        // Sum the vectors of each triplet, and create a new id for the triplet, which is the concatenation of the ids of the vectors
        JavaPairRDD<String, int[]> summed = joined2.mapToPair(x -> {
            int[] sum = IntStream.range(0, x._1._1._2.length)
                    .map(i -> x._1._1._2[i] + x._1._2._2[i] + x._2._2[i])
                    .toArray();
            return new Tuple2<>(x._1._1._1 + x._1._2._1 + x._2._1, sum);
        });

        // Compute the variance of each vector
        JavaPairRDD<String, Double> variance = summed.mapValues(x -> {
            // Compute the average
            double avg = Arrays.stream(x).average().getAsDouble();
            // Compute the variance
            double var = 0;
            for (int i : x) {
                var += Math.pow(i, 2);
            }
            var = var / x.length;
            return var - Math.pow(avg, 2);
        });

        // Collect the first 10 variances
        // Filter the variance RDD to only keep the triplets where the variance is smaller than the biggest tau
        
        if (lower) {
            variance = variance.filter(x -> x._2 < Arrays.stream(taus).max().getAsInt());
        } else {
            variance = variance.filter(x -> x._2 > Arrays.stream(taus).min().getAsInt());
        }

        variance.persist(StorageLevel.MEMORY_ONLY());
        
        for (double t : taus) {

            JavaPairRDD<String, Double> filtered;

            if (lower) {
                filtered  = variance.filter(x -> x._2 < t);
            } else {
                filtered  = variance.filter(x -> x._2 > t);
            }

            System.out.println("Tau " + t);

            System.out.printf("Number of triplets: %d\n", filtered.count());

            List<Tuple2<String, Double>> first10Variances = filtered.take(10);
    
            // Print the first 10 variances
            System.out.println("First 10 variances:");
            for (Tuple2<String, Double> f : first10Variances) {
                System.out.printf("%s: %.2f\n", f._1(), f._2());
            }  
        }      
        
        variance.unpersist();
            
    }


    // Main method which initializes a Spark context and runs the code for each question.
    // To skip executing a question while developing a solution, simply comment out the corresponding method call.
    public static void main(String[] args) {
        boolean onServer = false; // TODO: Set this to true if and only if building a JAR to run on the server

        JavaSparkContext sparkContext = getSparkContext(onServer);

        Dataset dataset = q1a(sparkContext, onServer);

        JavaRDD rdd = q1b(sparkContext, onServer);

//        // Get the time before executing the query
//        long startTime = System.nanoTime();
//        q2(sparkContext, dataset);
//        // Get the time after executing the query
//        long endTime = System.nanoTime();
//        // Print the time it took to execute the query
//        System.out.println("Time: " + (endTime - startTime) / 1000000 + " ms");

        // // Get the time before executing the query
        // long startTime2 = System.nanoTime();
        // q3(sparkContext, rdd);
        // // Get the time after executing the query
        // long endTime2 = System.nanoTime();
        // // Print the time it took to execute the query
        // System.out.println("Time: " + (endTime2 - startTime2) / 1000000 + " ms");

        Boolean lower = false;

        int[] taus = {};  
        double[] epsilon = {};
        double delta; 

        if (lower) {
            taus = new int[] {400};
            epsilon = new double[] {0.01, 0.001};
            delta = 0.1;             
        } else {
            taus = new int[] { 200000, 1000000};
            epsilon = new double[]  {0.0001, 0.001, 0.002, 0.01};
            delta = 0.1;    
        }

        for (double eps : epsilon) {
            System.out.println("Epsilon: " + eps + " Delta: " + delta);
            q4(sparkContext, rdd, taus, eps, delta, lower);
        }

        sparkContext.close();
    }
}
