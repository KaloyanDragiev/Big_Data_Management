import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.collection.Seq;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.IntStream;

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

        return sparkSession.read().csv(vectorsFilePath).repartition(sparkContext.defaultParallelism() * 4);
    }

    private static JavaRDD q1b(JavaSparkContext sparkContext, boolean onServer) {
        String vectorsFilePath = (onServer) ? "/vectors.csv" : "vectors.csv";

        return sparkContext.textFile(vectorsFilePath, 25);
    }

    private static void q2(JavaSparkContext sparkContext, Dataset dataset) {
        int[] taus = {20, 50, 310, 360, 410};

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

        splitted.persist();

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
        splitted.unpersist();
    }

    private static void q3(JavaSparkContext sparkContext, JavaRDD<String> rdd) {
        int[] taus = {20, 50, 310, 360, 410};

        System.out.println("Number of partitions: " + rdd.getNumPartitions());

        // Split every entry into an id and a vector
        JavaPairRDD<String, int[]> vectors = rdd.mapToPair(x -> {
            String[] split = x.split(",");
            int[] vector = Arrays.stream(split[1].split(";")).mapToInt(Integer::parseInt).toArray();
            return new Tuple2<>(split[0], vector);
        });

        vectors.persist(StorageLevel.MEMORY_ONLY());

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

        JavaPairRDD<String, Double> variance = summed.mapValues(x -> {
            // Compute the average
            double avg = 0; //Arrays.stream(x).average().getAsDouble();
            // Compute the variance
            double var = 0;
            for (int i : x) {
                avg += i;
                var += Math.pow(i, 2);
            }
            avg = avg / x.length;
            var = var / x.length;
            return var - Math.pow(avg, 2);
        });

        long start_time = System.nanoTime();

        // Only keep the triplets where the variance is smaller than the biggest tau
        JavaPairRDD<String, Double> filtered = variance.filter(x -> x._2 <= Arrays.stream(taus).max().getAsInt());

        filtered.persist(StorageLevel.MEMORY_ONLY());
        System.out.println("filtered: " + filtered.count() + ", time: " + (System.nanoTime() - start_time) / 1000000 + "ms");

        vectors.unpersist();

        for (int tau : taus) {
            System.out.println("Tau: " + tau);
            // Filter out the triplets where the variance is smaller than tau
            JavaPairRDD<String, Double> tauFiltered = filtered.filter(x -> x._2 <= tau);
            // Print the number of triplets
            System.out.println("Number of triplets: " + tauFiltered.count());
            // Print the all triplets
            tauFiltered.collect().stream().sorted(Comparator.comparing(Tuple2::_1)).forEach(x -> System.out.println(x._1() + ": " + x._2()));
        }

        filtered.unpersist();
    }

    public static int hash(int index, int numHashTables, int[] primes) {
        return Math.abs((primes[0] * index + primes[1]) % primes[2]) % numHashTables;
    }
    
    public static int[][] countMinSketch(int[] data, int numHashTables, int numHashFuncs) {
        // Initialize a list of prime numbers to be used for hashing
        int[][] primes = {{3553061, 3553049, 3552859}, {3553117, 3553139, 3553007}, {3554527, 3553453, 3555749}, {6790009, 2389213, 4390889}, {7888817, 8989837, 6791677}, {1889621, 2990017,8989837, }}; 
        
        
        // Initialize Count-Min sketch with given parameters
        int[][] sketch = new int[numHashFuncs][numHashTables];

        // Hash each element and increment corresponding counter
        for (int idx = 0; idx < data.length; idx++) {
            int element = data[idx];
            for (int hash_func = 0; hash_func < numHashFuncs; hash_func++){
               int hashVal = hash(idx, numHashTables, primes[hash_func]);
               sketch[hash_func][hashVal] += element; 
            }
        }

        return sketch;
    }

    public static double estimateVariance(int[][] sketch, int vectorLength) {
        // Create sketch dot product array
        long[] sketchDotProduct = new long[sketch.length];
        for (int i = 0; i < sketch.length; i++) {
            for (int j = 0; j < sketch[0].length; j++) {
                sketchDotProduct[i] += sketch[i][j] * sketch[i][j];
            }
        }
        // Get the min of sketch dot product array
        long dotProduct = Arrays.stream(sketchDotProduct).min().getAsLong();

        double mean = (double) Arrays.stream(sketch[0]).sum() / (double) vectorLength;

        // Estimate variance
        return ((double) dotProduct / vectorLength) - Math.pow(mean, 2);
    }

    private static void q4(JavaSparkContext sparkContext, JavaRDD<String> rdd, int[] taus, double eps, double delta, Boolean lower) {
        // Split every entry into an id and a vector
        JavaPairRDD<String, int[]> splitted = rdd.mapToPair(x -> {
            String[] split = x.split(",");
            int[] vector = Arrays.stream(split[1].split(";")).mapToInt(Integer::parseInt).toArray();
            return new Tuple2<>(split[0], vector);
        });

        // Take length of the vectors
        int vectorLength = splitted.take(1).get(0)._2.length;

        int w = (int) Math.ceil(Math.E / eps);
        int d = (int) Math.ceil(Math.log(1 / delta));

        // From splitted RDD, extract the data arrays and build Count-Min sketch for each data array together with its id
        JavaRDD<Tuple2<String, int[][]>> sketches = splitted.map(x -> new Tuple2<>(x._1, countMinSketch(x._2, w, d)));

        sketches.persist(StorageLevel.MEMORY_ONLY());

        // Join the RDD with itself to get all possible pairs, and filter out the pairs where the first id is smaller than the second
        JavaPairRDD<Tuple2<String, int[][]>, Tuple2<String, int[][]>> pairs =
                sketches.cartesian(sketches).filter(x -> x._1._1.compareTo(x._2._1) < 0);

        // Join it again with itself to get all possible triplets
        JavaPairRDD<Tuple2<Tuple2<String, int[][]>, Tuple2<String, int[][]>>, Tuple2<String, int[][]>> triplets =
                pairs.cartesian(sketches).filter(x -> x._1._2._1.compareTo(x._2._1) < 0);

        // Count the number of zeros in the sketches, and print the mean number of zeros
        triplets.mapValues(x -> {
            int zeros = 0;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < w; j++) {
                    if (x._2[i][j] == 0) {
                        zeros++;
                    }
                }
            }
            return zeros;
        }).values().collect().stream().mapToDouble(x -> x).average().ifPresent(System.out::println);

        // Print the number of triplets
        System.out.println("Number of triplets: " + triplets.count());

        // Sum the arrays from the triplets and append the keys
        JavaPairRDD<String, int[][]> sketchTriples = triplets.mapToPair(x -> {
            String key = x._1._1._1 + "," + x._1._2._1 + "," + x._2._1;
            int[][] sketch = new int[d][w];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < w; j++) {
                    sketch[i][j] = x._1._1._2[i][j] + x._1._2._2[i][j] + x._2._2[i][j];
                }
            }
            return new Tuple2<>(key, sketch);
        });

        JavaPairRDD<String, Double> sketchVariances = sketchTriples.mapValues(x -> estimateVariance(x, vectorLength));

        //Estimate variance of data stream using Count-Min sketch
        JavaPairRDD<String, Double> variance;

        if (lower) {
            variance = sketchVariances.filter(x -> x._2 < Arrays.stream(taus).max().getAsInt());
        } else {
            variance = sketchVariances.filter(x -> x._2 > Arrays.stream(taus).min().getAsInt());
        }

        variance.persist(StorageLevel.MEMORY_ONLY());
        System.out.println("Number of triplets after filter: " + variance.count());
        sketchTriples.unpersist();

        for (double t : taus) {

            JavaPairRDD<String, Double> filtered;

            if (lower) {
                filtered = variance.filter(x -> x._2 < t);

            } else {
                filtered = variance.filter(x -> x._2 > t);
            }

            System.out.println("Tau " + t);

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
        boolean onServer = true; // TODO: Set this to true if and only if building a JAR to run on the server

        JavaSparkContext sparkContext = getSparkContext(onServer);

        Dataset dataset = q1a(sparkContext, onServer);

        JavaRDD rdd = q1b(sparkContext, onServer);

        SparkContext ctx = SparkContext.getOrCreate(sparkContext.getConf());
        System.out.println("Number of cores: " + ctx.defaultParallelism());
        System.out.println("Number of workers: " + ctx.getExecutorMemoryStatus().size());

        // Get the time before executing the query
        long startTime = System.nanoTime();
        q2(sparkContext, dataset);
        // Get the time after executing the query
        long endTime = System.nanoTime();
        // Print the time it took to execute the query
        System.out.println("Time: " + (endTime - startTime) / 1000000 + " ms");

        // Get the time before executing the query
        System.out.println("Big vectors, q3 code");
        long startTime2 = System.nanoTime();
        q3(sparkContext, rdd);
        // Get the time after executing the query
        long endTime2 = System.nanoTime();
        // Print the time it took to execute the query
        System.out.println("Total time: " + (endTime2 - startTime2) / 1000000 + " ms");

        // Load the small vectors file, to cexecute the q3 code on the q2 dataset, and to run the q4 code
        // Change the path to the vectors_small.csv file if necessary
        String vectorsFilePath = (onServer) ? "/vectors_small.csv" : "vectors_small.csv";

        rdd = sparkContext.textFile(vectorsFilePath, 5);

        System.out.println("Small vectors, q3 code");
        long startTime2_2 = System.nanoTime();
        q3(sparkContext, rdd);
        // Get the time after executing the query
        long endTime2_2 = System.nanoTime();
        // Print the time it took to execute the query
        System.out.println("Total time: " + (endTime2_2 - startTime2_2) / 1000000 + " ms");

        System.out.println("q4");
        for (Boolean lower : new Boolean[] {true, false}) {
            System.out.println("Lower: " + lower);

            int[] taus = {};
            double[] epsilon = {};
            double delta;

            if (lower) {
                taus = new int[]{400};
                epsilon = new double[]{0.01, 0.001};
                delta = 0.1;
            } else {
                taus = new int[]{200000, 1000000};
                epsilon = new double[]{0.0001, 0.001, 0.002, 0.01};
                delta = 0.1;
            }

            for (double eps : epsilon) {
                long startTime3 = System.nanoTime();
                System.out.println("Epsilon: " + eps + " Delta: " + delta);
                q4(sparkContext, rdd, taus, eps, delta, lower);
                long endTime3 = System.nanoTime();
                System.out.println("Total time: " + (endTime3 - startTime3) / 1000000 + " ms");
            }
        }

        sparkContext.close();
    }
}
