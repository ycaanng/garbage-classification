package com.tianchi.garbage.classification;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class OfflineRunMain {
    public static void main(String[] args) throws Exception {
        String imageInputPath = System.getenv("IMAGE_INPUT_PATH");
        String imageModelPath = System.getenv("IMAGE_MODEL_PATH");
        String imageModelPackagePath = System.getenv("IMAGE_MODEL_PACKAGE_PATH");

        int[] inputShape = {1, 224, 224, 3};
        boolean ifReverseInputChannels = true;
//        float[] meanValues = {123.68f, 116.78f, 103.94f};
        float[] meanValues = {0.0f, 0.0f, 0.0f};
        float scale = 1.0f;
        String input = "input_1";
        boolean isTarGz = false;

        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);

        DataStream<IdLabel> stream;
        ImageDirSource source = new ImageDirSource();
        if (isTarGz) {
            stream = flinkEnv.addSource(source).setParallelism(1)
                    .flatMap(new PreprocessFlatMapFunction(inputShape)).setParallelism(2)
                    .flatMap(new PredictFlatMapFunction(imageModelPackagePath, inputShape, ifReverseInputChannels, meanValues, scale, input, isTarGz)).setParallelism(2);
        } else {
            stream = flinkEnv.addSource(source).setParallelism(1)
                    .flatMap(new PreprocessFlatMapFunction(inputShape)).setParallelism(2)
                    .flatMap(new PredictFlatMapFunction(imageModelPath, inputShape, ifReverseInputChannels, meanValues, scale, input, isTarGz)).setParallelism(2);
        }

        DataStream<Tuple2<String, Integer>> result = stream
                .flatMap(new CustomFlatMapFunction())
                .keyBy(0)
                .sum(1).setParallelism(1);

        result.print();
        flinkEnv.execute();
    }

    private static class CustomFlatMapFunction implements FlatMapFunction<IdLabel, Tuple2<String, Integer>> {
        private static final long serialVersionUID = -8354749620650810455L;

        @Override
        public void flatMap(IdLabel idLabel, Collector<Tuple2<String, Integer>> out) throws Exception {
            String predictLabel = idLabel.getLabel();
            String actualLabel = StringUtils.substring(idLabel.getId(), 0, StringUtils.lastIndexOf(idLabel.getId(), "_"));
            if (predictLabel.equals(actualLabel)) {
                out.collect(new Tuple2<>("true", 1));
            } else {
                out.collect(new Tuple2<>("false", 1));
            }
        }
    }
}
