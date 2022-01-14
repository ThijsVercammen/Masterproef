package com.example.onnx_runtime

import android.graphics.*
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer


const val DIM_BATCH_SIZE = 1;
const val DIM_PIXEL_SIZE = 3;
const val IMAGE_SIZE_X = 160;
const val IMAGE_SIZE_Y = 160;
const val IMAGE_MEAN: Float = 0.0f;
const val IMAGE_STD: Float = 1.0f;

fun preprocess(bitmap: Bitmap): FloatBuffer {
    val imgData = FloatBuffer.allocate(
        DIM_BATCH_SIZE
                * IMAGE_SIZE_X
                * IMAGE_SIZE_Y
                * DIM_PIXEL_SIZE)
    imgData.rewind()

    val bmpData = IntArray(IMAGE_SIZE_X * IMAGE_SIZE_Y)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    var idx: Int = 0
    for (i in 0..IMAGE_SIZE_X - 1) {
        for (j in 0..IMAGE_SIZE_Y - 1) {
            val pixelValue = bmpData[idx++]
            imgData.put(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            imgData.put(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            imgData.put(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        }
    }

    imgData.rewind()
    return imgData
}
/*
private fun YuvImage.toBitmap(): Bitmap? {
    val out = ByteArrayOutputStream()
    if (!compressToJpeg(Rect(0, 0, width, height), 100, out))
        return null
    val imageBytes: ByteArray = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

 */