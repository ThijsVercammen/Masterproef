package com.example.onnx_herkenning

//import com.example.onnx_runtime.R
//import com.example.onnx_runtime.MainActivity
import ai.onnxruntime.*
import android.Manifest
import android.content.Intent
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.*

const val DIM_BATCH_SIZE = 1;
const val DIM_PIXEL_SIZE = 3;
const val IMAGE_SIZE_X = 224;
const val IMAGE_SIZE_Y = 224;
const val IMAGE_MEAN: Float = 0.0f;
const val IMAGE_STD: Float = 1.0f;

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val buttonLoadImage = findViewById<View>(R.id.button) as Button
        val detectButton = findViewById<View>(R.id.detect) as Button

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), 1)
        }

        buttonLoadImage.setOnClickListener {
            val textView = findViewById<TextView>(R.id.result_text)
            textView.text = ""
            val i = Intent(
                Intent.ACTION_PICK,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            )
            startActivityForResult(i, RESULT_LOAD_IMAGE)
        }

        detectButton.setOnClickListener {
            println("INITIALIZE")
            var env = OrtEnvironment.getEnvironment()

            // Laad het correcte ONNX model
            var session = env.createSession(resources.openRawResource(R.raw.model_tf).readBytes())
            val labellist = loadLabelList()

            println("PREPARE DATA")
            val imageView = findViewById<View>(R.id.image) as ImageView
            var bitmap = (imageView.drawable as BitmapDrawable).bitmap

            // ONNX PyTorch input formaat
             //val shape = longArrayOf(1, 3, IMAGE_SIZE_X.toLong(), IMAGE_SIZE_Y.toLong())

            // ONNX Tensorflow input formaat
            val shape = longArrayOf(1, IMAGE_SIZE_X.toLong(), IMAGE_SIZE_Y.toLong(),3)

            bitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE_X, IMAGE_SIZE_Y, true)

            // Preprocess voor Pytorch ONNX model
             //val imgData = preprocessPy(bitmap)

            // Preprocess voor TensorFlow ONNX model
            val imgData = preprocessTf(bitmap)

            var t1 = OnnxTensor.createTensor(env, imgData, shape)

            println("EXECUTE")
            //var inputs = Collections.singletonMap("input", t1)
            var inputs = Collections.singletonMap("input_1", t1)
            val currentTime = Calendar.getInstance()
            val output = session?.run(inputs)
            println("INFERENCE TIME: " + (Calendar.getInstance()[Calendar.MILLISECOND] - currentTime[Calendar.MILLISECOND]))


            println("PROCESS OUTPUT")
            val labelVals = output?.get(0)?.value as Array<FloatArray>
            val probabilities = labelVals[0]

            var maxScore = -100000f
            var maxScoreIdx = -1
            for (i in 0 until probabilities.size) {
                if (probabilities[i] > maxScore) {
                    maxScore = probabilities[i]
                    maxScoreIdx = i
                }
            }
            val className = labellist?.get(maxScoreIdx)
            val textView = findViewById<TextView>(R.id.result_text)
            textView.text = className + "(" + maxScore + ")"

            println("FINISHED")
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            val selectedImage = data.data
            val filePathColumn = arrayOf(MediaStore.Images.Media.DATA)
            val cursor = contentResolver.query(
                selectedImage!!,
                filePathColumn, null, null, null
            )
            cursor!!.moveToFirst()
            val columnIndex = cursor.getColumnIndex(filePathColumn[0])
            val picturePath = cursor.getString(columnIndex)
            cursor.close()
            val imageView = findViewById<View>(R.id.image) as ImageView
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath))

            imageView.setImageURI(selectedImage)
        }
    }

    companion object {
        private const val RESULT_LOAD_IMAGE = 1
    }

    @Throws(IOException::class)
    private fun loadLabelList(): List<String>? {
        var labelList: List<String> = ArrayList()
        val reader = BufferedReader(InputStreamReader(this.assets.open("labels.txt")))
        labelList = reader.readLines()
        reader.close()
        return labelList
    }
}

fun preprocessPy(bitmap: Bitmap): FloatBuffer {
    val imgData = FloatBuffer.allocate(
        DIM_BATCH_SIZE
                * IMAGE_SIZE_X
                * IMAGE_SIZE_Y
                * DIM_PIXEL_SIZE
    )
    imgData.rewind()

    val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
    val bmpData = IntArray(IMAGE_SIZE_X * IMAGE_SIZE_Y)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    var idx: Int = 0
    for (i in 0..IMAGE_SIZE_X - 1) {
        for (j in 0..IMAGE_SIZE_Y - 1) {
            val idx = IMAGE_SIZE_Y * i + j
            val pixelValue = bmpData[idx]
            imgData.put(idx, (((pixelValue shr 16 and 0xFF) / 255f - 0.485f) / 0.229f))
            imgData.put(idx + stride, (((pixelValue shr 8 and 0xFF) / 255f - 0.456f) / 0.224f))
            imgData.put(idx + stride * 2, (((pixelValue and 0xFF) / 255f - 0.406f) / 0.225f))
        }
    }

    imgData.rewind()
    return imgData
}

fun preprocessTf(bitmap: Bitmap): FloatBuffer {
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


