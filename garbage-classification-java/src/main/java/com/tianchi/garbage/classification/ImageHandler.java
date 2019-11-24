package com.tianchi.garbage.classification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

public interface ImageHandler {
    JTensor getImageJTensor(byte[] imageBytes);
}
