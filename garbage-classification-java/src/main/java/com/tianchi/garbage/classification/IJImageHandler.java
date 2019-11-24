package com.tianchi.garbage.classification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import org.apache.zookeeper.server.ByteBufferInputStream;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IJImageHandler implements ImageHandler {
    private int[] inputShape;

    public IJImageHandler(int[] inputShape) {
        this.inputShape = inputShape;
    }

    @Override
    public JTensor getImageJTensor(byte[] imageBytes) {
        JTensor jTensor = null;
        try (InputStream in = new ByteBufferInputStream(ByteBuffer.wrap(imageBytes))) {
            BufferedImage bufferedImage = ImageIO.read(in);
            ColorProcessor colorProcessor = new ColorProcessor(bufferedImage);
            ImageProcessor resizeProcessor = colorProcessor.resize(inputShape[1], inputShape[2], true);

            int height = resizeProcessor.getHeight();
            int width = resizeProcessor.getWidth();
            int channels = resizeProcessor.getNChannels();

            List<Float> data = new ArrayList<>();
            List<Integer> shape = Arrays.asList(1, height, width, channels);

            List<Float> b = new ArrayList<>();
            List<Float> g = new ArrayList<>();
            List<Float> r = new ArrayList<>();

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int[] pixel = getPixel(resizeProcessor, w, h, channels);
                    r.add((float) pixel[0]);
                    g.add((float) pixel[1]);
                    b.add((float) pixel[2]);
                }
            }

            data.addAll(b);
            data.addAll(g);
            data.addAll(r);

            jTensor = new JTensor(data, shape);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return jTensor;
    }

    private int[] getPixel(ImageProcessor imageProcessor, int x, int y, int channels) {
        int[] pixel = new int[channels];
        imageProcessor.getPixel(x, y, pixel);
        return pixel;
    }
}
