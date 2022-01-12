package com.example.tensorflow_herkenning;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Trace;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.example.tensorflow_herkenning.ml.ModelHerkenningTf;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;

import java.io.IOException;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static int RESULT_LOAD_IMAGE = 1;

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

                System.out.println("PREPROCESS DATA");
                ImageView imageView = (ImageView) findViewById(R.id.image);
                Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
                List<Category> probability = null;

                try {
                    // Creates inputs for reference.
                    Trace.beginSection("recognizeImage");
                    TensorImage image = TensorImage.fromBitmap(bitmap);

                    System.out.println("INITIALISEER HERKENNER");

                    ModelHerkenningTf model = ModelHerkenningTf.newInstance(MainActivity.this);
                    // Runs model inference and gets result.
                    System.out.println("VOER HERKENNER UIT");
                    Calendar currentTime = Calendar.getInstance();
                    Trace.beginSection("runInference");
                    ModelHerkenningTf.Outputs outputs = model.process(image);
                    Trace.endSection();
                    System.out.println("INFERENCE TIME: " + (Calendar.getInstance().get(Calendar.SECOND) - currentTime.get(Calendar.SECOND)));
                    System.out.println("INFERENCE TIME: " + (Calendar.getInstance().get(Calendar.MILLISECOND) - currentTime.get(Calendar.MILLISECOND)));
                    probability = outputs.getProbabilityAsCategoryList();

                    System.out.println("VERWERK OUTPUT");

                    Category maxScore1 = null;
                    Category maxScore2 = null;
                    Category maxScore3 = null;
                    for (int i = 0; i < probability.size()-1; i++) {
                        if(maxScore1 == null) { maxScore1 = probability.get(i); }
                        else if(maxScore2 == null) { maxScore2 = probability.get(i); }
                        else if(maxScore3 == null) { maxScore3 = probability.get(i); }
                        else if (probability.get(i).getScore() > maxScore1.getScore()) {
                            maxScore3 = maxScore2;
                            maxScore2 = maxScore1;
                            maxScore1 = probability.get(i);
                        }
                        else if(probability.get(i).getScore() > maxScore2.getScore()) {
                            maxScore3 = maxScore2;
                            maxScore2 = probability.get(i);
                        }
                        else if(probability.get(i).getScore() > maxScore3.getScore()){
                            maxScore3 = probability.get(i);
                        }
                    }

                    TextView textView = findViewById(R.id.result_text);
                    textView.setText( "Top 3 resultaten: \n" +
                            maxScore1.getLabel() + " (" + maxScore1.getScore() + ") \n" +
                            maxScore2.getLabel() + " (" + maxScore2.getScore() + ") \n" +
                            maxScore3.getLabel() + " (" + maxScore3.getScore() + ") \n");
                    System.out.println("HERKENNER UITGEVOERD");
                    Trace.endSection();
                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }





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
}