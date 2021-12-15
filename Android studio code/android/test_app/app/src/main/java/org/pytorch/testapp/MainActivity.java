package org.pytorch.testapp;

//import static org.pytorch.testapp.CameraActivity.assetFilePath;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Map;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
//import org.pytorch.torchvision.BuildConfig;

public class MainActivity extends AppCompatActivity {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
    NativeLoader.loadLibrary("torchvision_ops");
  }

  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;
  private static int RESULT_LOAD_IMAGE = 1;


  private TextView mTextView;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

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
        // Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        Intent i = new Intent(Intent.ACTION_GET_CONTENT);
        i.setType("image/*");
        startActivityForResult(i, RESULT_LOAD_IMAGE);


      }
    });

    detectButton.setOnClickListener(new View.OnClickListener() {

      @Override
      public void onClick(View arg0) {

        Bitmap bitmap = null;
        Module transf = null;
        Module backbone = null;
        Module rpn = null;
        Module roi_heads_2 = null;
        Module roi_heads_3 = null;
        ArrayList<Integer> myList = new ArrayList<Integer>();

        //Getting the image from the image view
        ImageView imageView = (ImageView) findViewById(R.id.image);

        try {
          System.out.println("============================ initialise Model" +  "\n");
          //Read the image as Bitmap
          //bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
          bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
         // int w = bitmap.getWidth();
         // int h = bitmap.getHeight();
          //Here we reshape the image into 400*400
          bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);

          //Loading the model file.
          transf = Module.load(fetchModelFile(MainActivity.this, "transf_1.pt"));
          backbone = Module.load(fetchModelFile(MainActivity.this, "backbone.pt"));

          rpn = Module.load(fetchModelFile(MainActivity.this, "model_trace.pt"));
          roi_heads_2 = Module.load(fetchModelFile(MainActivity.this, "roi_h_2.pt"));
          roi_heads_3 = Module.load(fetchModelFile(MainActivity.this, "roi_h_3.pt"));
        } catch (IOException e) {
          finish();
        }


        //Input Tensor
        final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );
        //System.out.println("-------- " + input + " ------");

        //Calling the forward of the model to run our input
        long c[] = new long[3];
        c[0] = 3;
        c[1] = 400;
        c[2] = 400;
        Tensor b = Tensor.fromBlob(input.getDataAsFloatArray(), c);

        //Map<String, IValue> output1 = transf.forward(IValue.listFrom(b)).toDictStringKey();
        System.out.println("============================ Execute Model" +  "\n");
        IValue[] output2 = rpn.forward(IValue.listFrom(b)).toTuple();


        System.out.println("============================ Process Model" +  "\n");

        IValue[] det = output2[1].toList();
        System.out.println("============================ " + det.length + "\n");
        System.out.println("---------------------------------------------------------------------------------------" + "\n");
        Map<String, IValue> detections = det[0].toDictStringKey();

        for(String s: detections.keySet()){
          System.out.println("============================ Key: " + s + "\n");
        }
        Tensor result = detections.get("boxes").toTensor();
        float[] r = result.getDataAsFloatArray();
        /*
        for(int i = 0; i <= r.length; i=i+4){
          System.out.println("============================ " + r[i] + " - " + r[i+1] + " - " + r[i+2] + " - " + r[i+4] + "\n");
        }
        */

/*

                final float[] score_arr = output.getDataAsFloatArray();

                // Fetch the index of the value with maximum score
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                for (int i = 0; i < score_arr.length; i++) {
                    if (score_arr[i] > max_score) {
                        max_score = score_arr[i];
                        ms_ix = i;
                    }
                }
*/
        //Fetching the name from the list based on the index
        //String detected_class = ModelClasses.MODEL_CLASSES[ms_ix];

        //Writing the detected class in to the text view of the layout
        TextView textView = findViewById(R.id.result_text);
        //textView.setText(detected_class);
        System.out.println("-WORKED");
        textView.setText("WORKED");



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
      System.out.print("-------- " + imageView.getDrawable() + "\n");
      //Setting the URI so we can read the Bitmap from the image
      System.out.println("------------------- " + selectedImage);
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
}
