package com.example.tensorflow_yolo;

import android.content.res.AssetManager;

import java.io.IOException;

public class Yolov3Classifier extends Classifier {

    protected float mObjThresh = 0.1f;

    public Yolov3Classifier(AssetManager assetManager) throws IOException {
        super(assetManager, "model.tflite", "coco.txt", 416);
        mAnchors = new int[]{
                116,90,  156,198,  373,326, 10,13,  16,30,  33,23,  30,61,  62,45,  59,119
        };

        mMasks = new int[][]{{6,7,8},{3,4,5},{0,1,2}};
        mOutWidth = new int[]{52,26,13};
        mObjThresh = 0.6f;
    }

    @Override
    protected float getObjThresh() {
        return mObjThresh;
    }
}
