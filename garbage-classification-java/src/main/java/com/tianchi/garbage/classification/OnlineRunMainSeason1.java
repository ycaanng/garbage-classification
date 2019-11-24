package com.tianchi.garbage.classification;

import com.alibaba.tianchi.garbage_image_util.ImageClassSink;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class OnlineRunMainSeason1 {
    public static void main(String[] args) throws Exception {
        String imageInputPath = System.getenv("IMAGE_INPUT_PATH");
        String imageModelPath = System.getenv("IMAGE_MODEL_PATH");
        String imageModelPackagePath = System.getenv("IMAGE_MODEL_PACKAGE_PATH");

        int[] inputShape = {1, 224, 224, 3};
        boolean ifReverseInputChannels = true;
        float[] meanValues = {123.68f, 116.78f, 103.94f};
        float scale = 1.0f;
        String input = "input_1";
        boolean isTarGz = false;

        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);

        ImageDirSource source = new ImageDirSource();
        if (isTarGz) {
            flinkEnv.addSource(source).setParallelism(1)
                    .flatMap(new PreprocessFlatMapFunction(inputShape)).setParallelism(2)
                    .flatMap(new PredictFlatMapFunction(imageModelPackagePath, inputShape, ifReverseInputChannels, meanValues, scale, input, isTarGz)).setParallelism(2)
                    .addSink(new ImageClassSink()).setParallelism(1);
        } else {
            flinkEnv.addSource(source).setParallelism(1)
                    .flatMap(new PreprocessFlatMapFunction(inputShape)).setParallelism(2)
                    .flatMap(new PredictFlatMapFunction(imageModelPath, inputShape, ifReverseInputChannels, meanValues, scale, input, isTarGz)).setParallelism(2)
                    .addSink(new ImageClassSink()).setParallelism(1);
        }
        flinkEnv.execute();
    }
}
