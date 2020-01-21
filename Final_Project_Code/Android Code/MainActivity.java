package com.ece420.lab7;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.Manifest;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Objdetect;
import org.opencv.tracking.TrackerKCF;

import org.opencv.objdetect.CascadeClassifier ;
import org.opencv.core.Size ;
import org.opencv.core.Rect ;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static android.util.Config.LOGD;
import static org.opencv.imgproc.Imgproc.CV_CANNY_L2_GRADIENT;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;

import org.opencv.imgproc.Imgproc ;

import java.io.IOException;
import java.io.FileReader;
import java.io.FileReader ;
import java.util.Scanner;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    // UI Variables
    private Button controlButton;
    private Button identifyButton ;
    private Button edgeButton ;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    // Camera size
    private int myWidth;
    private int myHeight;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;

    // KCF Tracker variables
    private TrackerKCF myTacker;
    private Rect2d myROI = new Rect2d(0,0,0,0);
    private int myROIWidth = 70;
    private int myROIHeight = 70;
    private Scalar myROIColor = new Scalar(0,0,0);
    private int tracking_flag = -1;

    // Cascade Variable
    private CascadeClassifier Haar_Face_Casc ;

    private int haar_trigger = 0 ;
    private Rect[] facesArray ;
    private Bitmap bm ;

    private Mat cropCalc ;
    private Mat sub_mean ;
    private Mat av_vect ;

    private int detect = 0 ;

    private Mat train ;
    private Mat temp ;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // Request User Permission on Write/Read
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);}

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }


        // Setup control button
        controlButton = (Button)findViewById((R.id.controlButton));
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (v.getId() == R.id.controlButton && haar_trigger == 0) {
                    // Modify UI
                    controlButton.setText("STOP");
                    bm = null ;

                    // Set Haar_Trigger to Activate
                    haar_trigger = 1 ;
                }
                else {
                    // Modify UI
                    controlButton.setText("START");
                    haar_trigger = 0 ;

                    makeImage();

                }
            }
        });


        identifyButton = (Button)findViewById( (R.id.identifyButton) ) ;
        identifyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (v.getId() == R.id.identifyButton )
                {
                    // Need to setup Train Matrix
                    if (train == null )
                    {
                        train = new Mat(96*96,5,CvType.CV_32F ) ;

                        try {
                            InputStreamReader is = new InputStreamReader(getAssets()
                                    .open("train_face.csv"));

                            BufferedReader reader = new BufferedReader(is);
                            reader.readLine();

                            String line;
                            int row = 0 ;

                            while ((line = reader.readLine()) != null)
                            {
                                //Log.d("CREATION", line) ;
                                String[] listOf = line.split(",") ;

                                for ( int i = 0 ; i < listOf.length ; i++ )
                                {
                                    double var = Double.parseDouble(listOf[i]) ;
                                    train.put(row, i, var ) ;
                                }
                            }

                        } catch (Exception e) {

                        }

                        // Reading Sub-Mean Face CSV Fie
                        sub_mean = new Mat(96*96, 40, CvType.CV_32F ) ;
                        try {
                            InputStreamReader is = new InputStreamReader(getAssets()
                                    .open("sub_mean_face.csv"));

                            BufferedReader reader = new BufferedReader(is);
                            reader.readLine();

                            String line;
                            int row = 0 ;

                            while ((line = reader.readLine()) != null)
                            {
                                String[] listOf = line.split(",") ;

                                for ( int i = 0 ; i < listOf.length ; i++ )
                                {
                                    double var = Double.parseDouble(listOf[i]) ;
                                    sub_mean.put(row, i, var ) ;
                                }

                            }
                        } catch (Exception e) {

                        }

                        /// Reading from average vector
                        av_vect = new Mat(96*96, 1, CvType.CV_32F ) ;
                        try {
                            InputStreamReader is = new InputStreamReader(getAssets()
                                    .open("av_vect.csv"));

                            BufferedReader reader = new BufferedReader(is);
                            reader.readLine();

                            String line;
                            int row = 0 ;

                            while ((line = reader.readLine()) != null)
                            {
                                String[] listOf = line.split(",") ;

                                for ( int i = 0 ; i < listOf.length ; i++ )
                                {
                                    double var = Double.parseDouble(listOf[i]) ;
                                    av_vect.put(row, i, var ) ;
                                }

                            }
                        } catch (Exception e) {

                        }
                    }




                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            int person = recognition(cropCalc, train, sub_mean, av_vect );
                            TextView text = (TextView) findViewById(R.id.labelText);
                            text.setText( "Labeled Person: " + String.valueOf(person) );
                        }
                    });

                }
            }


        });


        edgeButton = (Button) findViewById( (R.id.edgeButton) ) ;
        edgeButton.setOnClickListener(new View.OnClickListener() {
              @Override
              public void onClick(View v) {
                    if (v.getId() == R.id.edgeButton )
                    {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Mat lines = new Mat() ;

                                Imgproc.Canny(temp, temp, 100, 200, 5, true  );
                                Imgproc.HoughLines(temp, lines, 2, 6.28 / 180, 100 );

                                double num_pair = lines.rows() ;
                                double result = (num_pair - 2000)/2000 ;


//                                for ( int i = 0 ; i < lines.rows() ; i++ )
//                                {
//                                    for ( int j = 0 ; j < lines.cols() ; j++ )
//                                    {
//                                        double [] thing = lines.get(i,j) ;
//
//                                        double r = thing[0] ;
//                                        double t = thing[1] ;
//
//                                        double xbar = Math.cos(t) ;
//                                        double ybar = Math.sin(t) ;
//
//                                        double X = r * xbar ;
//                                        double Y = r * ybar ;
//
//                                        int scale = 1000 ;
//
//
//
//                                        loc1 = [X + scale * (-ybar), Y + scale * xbar]
//                                        loc2 = [X - scale * (-ybar), Y - scale * xbar]
//
//
//                                    }
//                                }

                                Core.normalize(temp, temp, 0, 255, Core.NORM_MINMAX);
                                Core.convertScaleAbs(temp, temp);
                                temp.convertTo(temp, CvType.CV_8U );

                                Bitmap bmTemp = Bitmap.createBitmap(temp.cols(), temp.rows(), Bitmap.Config.ARGB_8888 );
                                Utils.matToBitmap(temp, bmTemp);

                                ImageView ig = (ImageView) findViewById(R.id.edgeResult );
                                ig.setImageBitmap(bmTemp);

                                TextView tv = (TextView) findViewById((R.id.resultThreat) ) ;


                                tv.setText( "Threat Level: " + String.format("%.2f", result )  );
                            }
                        });
                    }

              }
          });
        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        // mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();


        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);

        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    makeImage();
                    mOpenCvCameraView.enableView();

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public int recognition(Mat cropCalc, Mat eigenface, Mat sub_mean, Mat av_vect ){
        int num_face = 40 ;
        double min_error = 0 ;
        int person = -1 ;

        Mat resizeCrop =  new Mat(96, 96, CvType.CV_32F ) ;
        Size sz = new Size(96,96);
        Imgproc.resize(cropCalc, resizeCrop, sz );

        // Setting up the Matrix
        Mat weight_train = new Mat(eigenface.cols(), sub_mean.cols(), CvType.CV_32F ) ;
        Mat weight_test = new Mat (eigenface.cols(), 1, CvType.CV_32F ) ;
        Mat test_im_mean = new Mat(96*96, 1, CvType.CV_32F ) ;
        Mat diff_weight = new Mat(96*96, 1, CvType.CV_32F );

        // Histogram and Canny Edge
        resizeCrop.convertTo(resizeCrop, CvType.CV_8U );
        Imgproc.equalizeHist(resizeCrop, resizeCrop);
        Imgproc.Canny(resizeCrop, resizeCrop, 5,230 );
        resizeCrop.convertTo(resizeCrop, CvType.CV_32F ) ;

        // Obtain the Train and Test Weight
        Mat flattened = resizeCrop.reshape(1, 96*96 ) ;
        Core.subtract( flattened, av_vect, test_im_mean ) ;
        Core.gemm( eigenface.t(), sub_mean, 1, new Mat(), 0, weight_train ) ;
        Core.gemm( eigenface.t(), test_im_mean, 1, new Mat(), 0, weight_test ) ;


        /// Taking Norm basis or Euclidean Distance
        for(int j=0; j < num_face; j++ )
        {
            double error ;
            Scalar result ;

            Core.subtract( weight_train.col(j), weight_test, diff_weight) ;

            // Euclidean Distance Calc
            Core.multiply( diff_weight, diff_weight, diff_weight ) ;
            result = Core.sumElems(diff_weight) ;

            error = Math.sqrt( result.val[0] ) ;

            if ( error < min_error || min_error == 0 )
            {
                person = j ;
                min_error = error ;
            }

        }

        person = person/10  ;

        return(person);
    }

    // Helper Function to map single integer to color scalar
    // https://www.particleincell.com/2014/colormap/
    public void setColor(int value) {
        double a=(1-(double)value/100)/0.2;
        int X=(int)Math.floor(a);
        int Y=(int)Math.floor(255*(a-X));
        double newColor[] = {0,0,0};
        switch(X)
        {
            case 0:
                // r=255;g=Y;b=0;
                newColor[0] = 255;
                newColor[1] = Y;
                break;
            case 1:
                // r=255-Y;g=255;b=0
                newColor[0] = 255-Y;
                newColor[1] = 255;
                break;
            case 2:
                // r=0;g=255;b=Y
                newColor[1] = 255;
                newColor[2] = Y;
                break;
            case 3:
                // r=0;g=255-Y;b=255
                newColor[1] = 255-Y;
                newColor[2] = 255;
                break;
            case 4:
                // r=Y;g=0;b=255
                newColor[0] = Y;
                newColor[2] = 255;
                break;
            case 5:
                // r=255;g=0;b=255
                newColor[0] = 255;
                newColor[2] = 255;
                break;
        }
        myROIColor.set(newColor);
        return;
    }

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        myWidth = width;
        myHeight = height;
        myROI = new Rect2d(myWidth / 2 - myROIWidth / 2,
                            myHeight / 2 - myROIHeight / 2,
                            myROIWidth,
                            myROIHeight);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    public void makeImage() {
        if ( bm != null  )
        {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    ImageView iv = (ImageView) findViewById(R.id.cropFace);
                    iv.setImageBitmap(bm);
                }
            });
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Timer
        long start = Core.getTickCount();

        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();

        // Grab camera frame in gray format
        mGray = inputFrame.gray();


        if ( haar_trigger == 1 )
        {
            String path_file = "/storage/emulated/0/DCIM/haarcascade_frontalface_default.xml";
            Haar_Face_Casc = new CascadeClassifier(path_file);

            Mat test = new Mat();

            cvtColor(mRgba, test, Imgproc.COLOR_RGB2BGR);
            MatOfRect rec = new MatOfRect();

            Haar_Face_Casc.detectMultiScale(test, rec, 1.1, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size(1020, 1020));
            facesArray = rec.toArray();

            if (facesArray.length > 0 )
            {
                for (int i = 0; i < 1; i++) {
                    Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);

                    int top_left_x = (int) facesArray[i].tl().x;
                    int top_left_y = (int) facesArray[i].tl().y;

                    int bottom_right_x = (int) facesArray[i].br().x;
                    int bottom_right_y = (int) facesArray[i].br().y;


                    int total_row = bottom_right_y - top_left_y ;
                    int total_col = bottom_right_x - top_left_x ;

                    Mat crop = new Mat(total_row, total_col, CvType.CV_8U );
                    cropCalc = new Mat(total_row, total_col, CvType.CV_32F );


                    for (int row = top_left_y ; row <  bottom_right_y ; row++) {
                        for (int col = top_left_x ; col < bottom_right_x ; col++) {
                            crop.put(row - top_left_y,col - top_left_x, mRgba.get(row,col) ) ;
                            cropCalc.put(row - top_left_y,col - top_left_x, mRgba.get(row,col) ) ;
                        }
                    }

                    bm = Bitmap.createBitmap(crop.cols(), crop.rows(), Bitmap.Config.ARGB_8888 );
                    Utils.matToBitmap(crop, bm);
                }

            }
        }

        temp = mGray.clone() ;



        return mRgba;
    }
}