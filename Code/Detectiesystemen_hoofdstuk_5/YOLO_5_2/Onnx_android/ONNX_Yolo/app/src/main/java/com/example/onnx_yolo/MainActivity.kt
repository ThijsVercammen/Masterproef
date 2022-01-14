package com.example.onnx_yolo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.Manifest
import android.content.Intent
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val buttonLoadImage = findViewById<View>(R.id.button) as Button
        val detectButton = findViewById<View>(R.id.detect) as Button
        val mHandler = Handler()

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
            println("INITIALIZE DETECTOR")
            var env = OrtEnvironment.getEnvironment()
            var session = env.createSession(resources.openRawResource(R.raw.model_3).readBytes())
            val labellist = loadLabelList()

            println("PREPARE DATA")
            val imageView = findViewById<View>(R.id.image) as ImageView
            var bitmap = (imageView.drawable as BitmapDrawable).bitmap
            val shape = longArrayOf(1, 416, 416, 3)
            bitmap = Bitmap.createScaledBitmap(bitmap, 416, 416, true)
            val imgData = preprocess(bitmap)
            var t1 = OnnxTensor.createTensor(env, imgData, shape)

            println("EXECUTE DETECTOR")
            var inputs = Collections.singletonMap("input_1", t1)
            val output = session.run(inputs)
            output.use{
                println("PROCCESS OUTPUT")
                /*
                val boxes = output.get(0)?.value as Array<FloatArray>
                val labels = output.get(1)?.value as IntArray
                val scores = output.get(2)?.value as FloatArray
                val num_det = labels.size
                */

                val boxes = output.get("keras_layer_1")?.get()?.value as Array<Array<FloatArray>>
                val labels = output.get("keras_layer_2")?.get()?.value as Array<FloatArray>
                val scores = output.get("keras_layer_4")?.get()?.value as Array<FloatArray>
                val num_det = (output.get("keras_layer_5")?.get()?.value as FloatArray)[0].toInt()

                val recognitions = mutableListOf<Recognition>()
                for (i in 0..num_det-1) {
                    if (scores[0][i] >= 0.85) {
                        val detection = RectF(
                            boxes[0][i][1] * 160,
                            boxes[0][i][0] * 160,
                            boxes[0][i][3] * 160,
                            boxes[0][i][2] * 160
                        )
                        recognitions.add(
                            Recognition(
                                "" + i, "" + (labellist?.get(labels[0][i].toInt() - 1)),
                                scores[0][i],
                                detection
                            )
                        )
                    }
                    /*
                   if (scores[i] >= 0.85) {
                       val rect = RectF(
                           boxes[i][1] * 160,
                           boxes[i][0] * 160,
                           boxes[i][3] * 160,
                           boxes[i][2] * 160
                       )
                       recognitions.add(Recognition("" + i,"" + labels[i], scores[i], rect))
                   }

                     */
                }
                println("DRAW BOXES")
                bitmap = Bitmap.createBitmap(bitmap)
                val canvas = Canvas(bitmap)
                val paint = Paint()
                println(recognitions.size)
                paint.setColor(Color.RED)
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = 2.0f
                var ra = ""
                for (result in recognitions) {
                    val location = result.location
                    if (location != null && result.confidence >= 0.1) {
                        canvas.drawRect(location, paint)
                        ra += result.title.toString() + " (" + result.confidence + "%)" + "\n"
                    }
                }
                val textView = findViewById<TextView>(R.id.result_text)
                textView.text = ra
            }

            mHandler.post(Runnable { imageView.setImageBitmap(bitmap) })
            session.close()
            env.close()
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
        val reader = BufferedReader(InputStreamReader(this.assets.open("coco.txt")))
        labelList = reader.readLines()
        reader.close()
        return labelList
    }
}