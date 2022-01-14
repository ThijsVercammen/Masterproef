package org.pytorch.testapp;

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
import android.os.Handler;
import android.provider.MediaStore;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;
    Bitmap bitmap = null;
    List<Integer> colorList = new ArrayList<Integer>();

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("pytorch_jni");
        NativeLoader.loadLibrary("torchvision_ops");
    }

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

                Module model = null;
                Handler mHandler = new Handler();
                List<String> labelList = null;

                try {
                    System.out.println("INITSIALISE DETECTOR");
                    model = Module.load(fetchModelFile(MainActivity.this, "model.pt"));
                    labelList = loadLabelList();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                colorList.add(Color.RED);
                colorList.add(Color.GREEN);
                colorList.add(Color.BLUE);
                colorList.add(Color.YELLOW);
                colorList.add(Color.BLACK);
                colorList.add(Color.WHITE);
                colorList.add(Color.CYAN);
                colorList.add(Color.GRAY);
                colorList.add(Color.MAGENTA);
                colorList.add(Color.DKGRAY);

                System.out.println("PREPROCESS INPUT DATA");
                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);
                bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
                bitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true);

                //Input Tensor
                float[] mean = new float[]{0.0f, 0.0f, 0.0f};
                float[] std = new float[]{1.0f, 1.0f, 1.0f};
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std);

                //Calling the forward of the model to run our input
                long[] shape = new long[]{3, 160, 160};
                //shape[0] = 3; shape[1] = 160; shape[2] = 160;
                Tensor b = Tensor.fromBlob(input.getDataAsFloatArray(), shape);

                //execute model
                System.out.println("EXCECUTE MODEL");
                IValue[] output = model.forward(IValue.listFrom(b)).toTuple();

                //read outputs
                System.out.println("PROCESS OUTPUT");
                IValue[] det = output[1].toList();
                Map<String, IValue> detections = det[0].toDictStringKey();

                Tensor boxes = detections.get("boxes").toTensor();
                Tensor scores = detections.get("scores").toTensor();
                Tensor labels = detections.get("labels").toTensor();

                final float[] boxesData = boxes.getDataAsFloatArray();
                final float[] scoresData = scores.getDataAsFloatArray();
                final long[] labelData = labels.getDataAsLongArray();

                //get bounding box location and labels above the confidence treshold 0.8
                ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
                Map<String, Integer> colorMap = new HashMap<>();
                int index = 0;

                for (int i = 0; i < scoresData.length; i++) {
                    if (scoresData[i] >= 0.8) {
                        RectF rect = new RectF(
                                boxesData[4 * i + 0],
                                boxesData[4 * i + 1],
                                boxesData[4 * i + 2],
                                boxesData[4 * i + 3]);
                        String label = labelList.get((int) labelData[i]-1);
                        if(!colorMap.containsKey(label)){
                            colorMap.put(label, colorList.get(index));
                            index++;
                        }
                        recognitions.add(new Recognition("" + i, label, scoresData[i], rect, colorMap.get(label)));
                    }
                }

                // Draw Bounding boxes
                System.out.println("DRAW BOXES");
                //Writing the detected class in to the text view of the layout
                TextView textView = findViewById(R.id.result_text);
                bitmap = Bitmap.createBitmap(bitmap);
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                final Canvas canvas = new Canvas(bitmap);
                final Paint paint = new Paint();
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2.0f);
                textView.setText("");

                for (final Recognition result : recognitions) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= 0.1) {
                        paint.setColor(result.getColor());
                        canvas.drawRect(location, paint);
                        Spannable word = new SpannableString(result.getTitle() + " (" + result.getConfidence() + "%)" + "\n");
                        word.setSpan(new ForegroundColorSpan(result.getColor()), 0, word.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                        textView.append(word);
                    }
                }

                mHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        imageView.setImageBitmap(bitmap);
                    }
                });

                System.out.println("FINISHED DETECTION");
                model.destroy();
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
            imageView.setImageURI(selectedImage);


        }
    }

    public static String fetchModelFile(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
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

    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
}
