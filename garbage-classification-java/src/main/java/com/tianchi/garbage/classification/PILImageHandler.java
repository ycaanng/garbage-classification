package com.tianchi.garbage.classification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.cvGet2D;

public class PILImageHandler implements ImageHandler {
    private int[] inputShape;

    public PILImageHandler(int[] inputShape) {
        this.inputShape = inputShape;
    }

    @Override
    public JTensor getImageJTensor(byte[] imageBytes) {
        opencv_core.IplImage originImage = new opencv_core.IplImage(opencv_imgcodecs.imdecode(new opencv_core.Mat(imageBytes, true), -1));
        opencv_core.IplImage resizedImage = opencv_core.IplImage.create(inputShape[1], inputShape[2], originImage.depth(), inputShape[3]);
        opencv_imgproc.cvResize(originImage, resizedImage, opencv_imgproc.CV_INTER_LINEAR);

        int height = resizedImage.height();
        int width = resizedImage.width();
        int channels = resizedImage.nChannels();

        List<Float> data = new ArrayList<>();
        List<Integer> shape = Arrays.asList(1, height, width, channels);

        List<Float> r = new ArrayList<>();
        List<Float> b = new ArrayList<>();
        List<Float> g = new ArrayList<>();

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                Float[] pixel = getPixel(resizedImage, h, w);
                r.add(pixel[0]);
                g.add(pixel[1]);
                b.add(pixel[2]);
            }
        }

        data.addAll(b);
        data.addAll(g);
        data.addAll(r);

        return new JTensor(data, shape);
    }

    private Float[] getPixel(opencv_core.IplImage image, int r, int c) {
        opencv_core.CvScalar pixel = cvGet2D(image, r, c);
        return Arrays.asList((float) pixel.val(0), (float) pixel.val(1), (float) pixel.val(2)).toArray(new Float[0]);
    }
}
