package com.example.pytorch_herkenning;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Calendar;

public class MainActivity extends AppCompatActivity {

    private static int RESULT_LOAD_IMAGE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]  {android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        buttonLoadImage.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");
                Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                i.setType("image/*");
                startActivityForResult(i, RESULT_LOAD_IMAGE);
            }
        });

        detectButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {

                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);
                Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                Module module = null;
                try {
                    module = Module.load(assetFilePath(MainActivity.this, "model.pt"));
                } catch (IOException e) {
                    System.out.println("FAILED");
                    e.printStackTrace();
                }
                System.out.println(bitmap.getHeight());
                bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

                //Input Tensor
               // float[] mean = new float[]{0.0f, 0.0f, 0.0f};
               // float[] std = new float[]{1.0f, 1.0f, 1.0f};
               // final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std);
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                Calendar currentTime = Calendar.getInstance();
                Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
                System.out.println("INFERENCE TIME: " + (Calendar.getInstance().get(Calendar.MILLISECOND) - currentTime.get(Calendar.MILLISECOND)));

                float[] scores = outputTensor.getDataAsFloatArray();

                float maxScore = -Float.MAX_VALUE;
                int maxScoreIdx = -1;
                for (int i = 0; i < scores.length; i++) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i];
                        maxScoreIdx = i;
                    }
                }
                String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

                TextView textView = findViewById(R.id.result_text);
                System.out.println("res");
                textView.setText(className + " (" +maxScore+")");
            }
        });

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));
            System.out.println(imageView);
            imageView.setImageURI(selectedImage);
        }
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}