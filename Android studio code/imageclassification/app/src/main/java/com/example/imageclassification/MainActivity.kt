package com.example.imageclassification

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.example.imageclassification.ml.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.task.vision.detector.Detection

import org.tensorflow.lite.task.vision.detector.ObjectDetector

import org.tensorflow.lite.task.core.BaseOptions

import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions




class MainActivity : AppCompatActivity() {

    lateinit var select_image_button : Button
    lateinit var make_prediction : Button
    lateinit var img_view : ImageView
    lateinit var text_view : TextView
    lateinit var bitmap: Bitmap
    lateinit var camerabtn : Button

    private val channelSize = 3
    var inputImageWidth = 1
    var inputImageHeight = 1
    private var modelInputSize = inputImageWidth * inputImageHeight * channelSize
    val resultArray = Array(8) { ByteArray(3) }

    public fun checkandGetpermissions(){
        if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        }
        else{
            Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
        }/*
        if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        }
        else{
            Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
        }*/
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == 100){
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
            }
            else{
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        select_image_button = findViewById(R.id.button)
        make_prediction = findViewById(R.id.button2)
        img_view = findViewById(R.id.imageView2)
        text_view = findViewById(R.id.textView)
        camerabtn = findViewById<Button>(R.id.camerabtn)

        // handling permissions
        checkandGetpermissions()

        val labels = application.assets.open("labels_1.txt").bufferedReader().use { it.readText() }.split("\n")

        select_image_button.setOnClickListener(View.OnClickListener {
            Log.d("mssg", "button pressed")
            var intent : Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 250)
        })

        make_prediction.setOnClickListener(View.OnClickListener {
            var resized = Bitmap.createScaledBitmap(bitmap, 160, 160, true)
            val model = Model1.newInstance(this)
            var tbuffer = TensorImage.fromBitmap(bitmap)
            var byteBuffer = tbuffer.buffer
            File f = new File()
            Interpreter interpreter = new Interpreter()
            //////////////////////////////////////
            //val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 300, 1, 3), DataType.FLOAT32)
            //inputFeature0.loadBuffer(byteBuffer)
            /*
            val options = ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().build())
                .setMaxResults(1)
                .build()
            val objectDetector = ObjectDetector.createFromFileAndOptions(
                this, "metadata.tflite", options
            )

// Run inference

// Run inference
            val results: List<Detection> = objectDetector.detect(tbuffer)
// Runs model inference and gets result.
            //val outputs = model.process(inputFeature0)
            System.out.println("111111111111111111")
            val image = TensorImage.fromBitmap(bitmap)
            val mod = Metadata.newInstance(this)
// Runs model inference and gets result.
            val output = mod.process(image)
            System.out.println("111111111111111111")
            val detectionResult = output.detectionResultList.get(0)
            System.out.println("111111111111111111")
// Gets result from DetectionResult.
            val location = detectionResult.locationAsRectF;
            System.out.println("-------------" + location)
            val category = detectionResult.categoryAsString;
            System.out.println("-------------" + category)
            val score = detectionResult.scoreAsInt;
            System.out.println("-------------" + score)

// Releases model resources if no longer used.
            model.close()



            model.close()

*/
// Runs model inference and gets result.
            val outputs = model.process(tbuffer)
            val outputFeature0 = outputs.probabilityAsCategoryList

            if(outputFeature0[0].score > outputFeature0[1].score){
                text_view.setText("CAT");
            } else {
                text_view.setText("DOG");
            }

// Releases model resources if no longer used.
            model.close()


        })

        camerabtn.setOnClickListener(View.OnClickListener {
            var camera : Intent = Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(camera, 200)
        })


    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 250){
            img_view.setImageURI(data?.data)

            var uri : Uri ?= data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
        }
        else if(requestCode == 200 && resultCode == Activity.RESULT_OK){
            bitmap = data?.extras?.get("data") as Bitmap
            img_view.setImageBitmap(bitmap)
        }

    }
}