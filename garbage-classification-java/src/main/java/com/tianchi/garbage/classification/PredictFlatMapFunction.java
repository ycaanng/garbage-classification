package com.tianchi.garbage.classification;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PredictFlatMapFunction extends RichFlatMapFunction<Tuple2<String, JTensor>, IdLabel> {
    private static final long serialVersionUID = 171327082284055788L;

    private String imageModelPath;
    private int[] inputShape;
    private boolean ifReverseInputChannels;
    private float[] meanValues;
    private float scale;
    private String input;
    private boolean isTarGz;

    private ImageClassificationModel imageClassificationModel;
    private Map<Integer, String> dict;

    public PredictFlatMapFunction(String imageModelPath, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale, String input, boolean isTarGz) {
        this.imageModelPath = imageModelPath;
        this.inputShape = inputShape;
        this.ifReverseInputChannels = ifReverseInputChannels;
        this.meanValues = meanValues;
        this.scale = scale;
        this.input = input;
        this.isTarGz = isTarGz;
    }

    @Override
    public void flatMap(Tuple2<String, JTensor> item, Collector<IdLabel> out) throws Exception {
        List<List<JTensor>> inputs = Collections.singletonList(Collections.singletonList(item.f1));
        float[] softMaxProb = imageClassificationModel.doPredict(inputs).get(0).get(0).getData();
        int index = getMaxProb(softMaxProb);
        String label = dict.get(index);
        out.collect(new IdLabel(item.f0, label));
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        imageClassificationModel = new ImageClassificationModel();
        if (isTarGz) {
            byte[] imageModelBytes = Files.readAllBytes(Paths.get(imageModelPath));
            imageClassificationModel.doLoadTF(imageModelBytes, inputShape, ifReverseInputChannels, meanValues, scale, input);
        } else {
            imageClassificationModel.doLoadTF(imageModelPath, inputShape, ifReverseInputChannels, meanValues, scale, input);
        }
        dict = getDict();
    }

    @Override
    public void close() throws Exception {
        if (imageClassificationModel != null) {
            imageClassificationModel.release();
        }
    }

    private static class ImageClassificationModel extends AbstractInferenceModel {
        private static final long serialVersionUID = 3075001631913814962L;

        public ImageClassificationModel() {
            super();
        }
    }

    private Map<Integer, String> getDict() {
        Map<Integer, String> dict = new HashMap<>();
        try (InputStream in = PredictFlatMapFunction.class.getClassLoader().getResourceAsStream("class_index.txt")) {
            LineIterator lineIterator = IOUtils.lineIterator(in, StandardCharsets.UTF_8);
            while (lineIterator.hasNext()) {
                String line = lineIterator.next();
                String[] split = line.split(" ");
                dict.put(Integer.parseInt(split[1]), split[0]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dict;
    }

    private int getMaxProb(float[] softMaxProb) {
        float temp = softMaxProb[0];
        int index = 0;
        for (int i = 0; i < softMaxProb.length; i++) {
            if (softMaxProb[i] > temp) {
                index = i;
                temp = softMaxProb[i];
            }
        }
        return index;
    }
}
