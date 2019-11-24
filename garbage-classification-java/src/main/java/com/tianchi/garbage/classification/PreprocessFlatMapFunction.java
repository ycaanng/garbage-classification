package com.tianchi.garbage.classification;

import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

public class PreprocessFlatMapFunction extends RichFlatMapFunction<ImageData, Tuple2<String, JTensor>> {
    private static final long serialVersionUID = -1963469614070121448L;

    private int[] inputShape;

    public PreprocessFlatMapFunction(int[] inputShape) {
        this.inputShape = inputShape;
    }

    @Override
    public void flatMap(ImageData imageData, Collector<Tuple2<String, JTensor>> out) throws Exception {
        String id = imageData.getId();
        byte[] buffer = imageData.getImage();

        ImageHandler imageHandler = new IJImageHandler(inputShape);
        JTensor jTensor = imageHandler.getImageJTensor(buffer);

        out.collect(new Tuple2<>(id, jTensor));
    }

    @Override
    public void open(Configuration parameters) throws Exception {

    }
}
