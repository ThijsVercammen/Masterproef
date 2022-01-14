package com.example.tensorflow_yolo;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private static int RESULT_LOAD_IMAGE = 1;

    private Handler mHandler = new Handler();

    private ByteBuffer imgData = null;
    private int DIM_IMG_SIZE_X = 416;
    private int DIM_IMG_SIZE_Y = 416;
    private int DIM_PIXEL_SIZE = 3;
    Bitmap res = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
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
                Yolov3Classifier classifier = null;
                try {
                    classifier = new Yolov3Classifier(getAssets());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                ImageView imageView = (ImageView) findViewById(R.id.image);
                Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
                res = processBitmap(bitmap,classifier.getInputSize());

                ArrayList<Classifier.Recognition> results = null;

                results = classifier.RecognizeImage(res);

                res = Bitmap.createBitmap(res);
                final Canvas canvas = new Canvas(res);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2.0f);
                String ra = "";

                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= 0.1) {
                        canvas.drawRect(location, paint);
                        ra += result.getTitle() + " (" + result.getConfidence() + "%)" + "\n";
                    }
                }

                mHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        imageView.setImageBitmap(res);
                    }
                });

                TextView textView = findViewById(R.id.result_text);
                textView.setText(ra);
                System.out.println("FINISHED DETECTION");
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = {MediaStore.Images.Media.DATA};

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(selectedImage);
        }
    }

    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }

    public static Bitmap processBitmap(Bitmap source,int size){

        int image_height = source.getHeight();
        int image_width = source.getWidth();

        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);

        Matrix frameToCropTransformations = getTransformationMatrix(image_width,image_height,size,size,0,false);
        Matrix cropToFrameTransformations = new Matrix();
        frameToCropTransformations.invert(cropToFrameTransformations);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(source, frameToCropTransformations, null);

        return croppedBitmap;


    }
}