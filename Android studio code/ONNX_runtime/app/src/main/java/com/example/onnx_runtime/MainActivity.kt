package com.example.onnx_runtime

import ai.onnxruntime.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.onnx_runtime.R
import android.os.Build
import android.widget.TextView
import android.content.Intent
import android.provider.MediaStore
import com.example.onnx_runtime.MainActivity
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.Manifest
import android.app.Activity
import android.graphics.BitmapFactory
import android.view.View
import android.widget.Button
import android.widget.ImageView
import java.io.IOException
import java.nio.FloatBuffer
import java.util.*

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
            /*
            var bitmap: Bitmap? = null
            var env : OrtEnvironment? = null
            //Getting the image from the image view
            val imageView = findViewById<View>(R.id.image) as ImageView
            bitmap = (imageView.drawable as BitmapDrawable).bitmap

            //Here we reshape the image into 400*400
            bitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true)
            val imgData = preProcess(bitmap)
            val shape = longArrayOf(1, 3, 160, 160)

            env = OrtEnvironment.getEnvironment()
            val a = resources.openRawResource(R.raw.model_lite1).readBytes()
            val session: OrtSession? = env?.createSession(a)
            val inputName = session?.inputNames?.iterator()?.next()
            println("LOADED")
            env.use {
                val tensor = OnnxTensor.createTensor(env, imgData, shape)
                tensor.use {
                    val output = session?.run(Collections.singletonMap(inputName, tensor))
                    output.use {
                        println("---------- " + (output?.get(0)?.value))
                    }
                }
            }
*/
            val imageView = findViewById<View>(R.id.image) as ImageView
            var bitmap = (imageView.drawable as BitmapDrawable).bitmap
            val shape = longArrayOf(1, 160, 160, 3)
            //Here we reshape the image into 400*400
            bitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true)
            val imgData = preprocess(bitmap)
            var env = OrtEnvironment.getEnvironment()
            var session = env.createSession(resources.openRawResource(R.raw.model_2).readBytes())
            var t1 = OnnxTensor.createTensor(env, imgData, shape)

            var inputs = Collections.singletonMap("input_1", t1)
            val output = session.run(inputs)
            //val result = session.run(inputs)
            output.use{
                val a = output.get("keras_layer_1")?.get()?.value as Array<Array<FloatArray>>
                println("---------- " + a[0][0][0])
            }


            //final float[] score_arr = output.getDataAsFloatArray();

            // Fetch the index of the value with maximum score
            val textView = findViewById<TextView>(R.id.result_text)
            textView.text = "WORKED"
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


            //imageView.setImageURI(null);
            imageView.setImageURI(selectedImage)
        }
    }

    companion object {
        private const val RESULT_LOAD_IMAGE = 1
    }


}
internal data class Result(
    var detectedIndices: List<Int> = emptyList(),
    var detectedScore: MutableList<Float> = mutableListOf<Float>(),
    var processTimeMs: Long = 0
) {}