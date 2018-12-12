package com.alensalihbasic.wreckfacejavacv;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ToggleButton;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class AgeRecognizer extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "AgeRecognizer";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Net mAgeNet;
    private CascadeClassifier mFaceDetector;
    private File mCascadeFile;
    private Mat mRgba, mGray;
    private int mAbsoluteFaceSize = 0;
    private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"};
    private int mCameraId = 1;

    //Connection between app and OpenCV Manager
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    new AsyncTask<Void, Void, Void>() {
                        @Override
                        protected Void doInBackground(Void... voids) {
                            try {
                                //Loading detection classifier from resources
                                InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                                File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                                mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                                FileOutputStream os = new FileOutputStream(mCascadeFile);

                                byte[] buffer = new byte[4096];
                                int bytesRead;
                                while ((bytesRead = is.read(buffer)) != -1) {
                                    os.write(buffer, 0, bytesRead);
                                }
                                is.close();
                                os.close();

                                mFaceDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                                if (mFaceDetector.empty()) {
                                    Log.e(TAG, "Failed to load cascade classfier");
                                    mFaceDetector = null;
                                } else {
                                    Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                                }
                                cascadeDir.delete();

                            } catch (IOException e) {
                                e.printStackTrace();
                                Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                            }
                            return null;
                        }
                    }.execute();

                    mOpenCvCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
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
    protected void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recognizer_age);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.setCvCameraViewListener(this);

        ToggleButton mFlipCamera = findViewById(R.id.toggle_camera);
        mFlipCamera.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mCameraId = 0;
                    mOpenCvCameraView.disableView();
                    mOpenCvCameraView.setCameraIndex(mCameraId);
                    mOpenCvCameraView.enableView();
                } else {
                    mCameraId = 1;
                    mOpenCvCameraView.disableView();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                }
            }
        });

        Button mBackButton = findViewById(R.id.back_button);
        mBackButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Methods.reset();
                Intent backIntent = new Intent(AgeRecognizer.this, WelcomeScreen.class);
                startActivity(backIntent);
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();

        //Loading Caffe model to Dnn
        String proto = getPath("deploy_age.prototxt", this);
        String weights = getPath("age_net.caffemodel", this);
        mAgeNet = Dnn.readNetFromCaffe(proto, weights);

        if (mAgeNet.empty()) {
            Log.i(TAG, "Network loading failed");
        } else {
            Log.i(TAG, "Network loading success");
        }
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        //Computing absolute face size
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            float mRelativeFaceSize = 0.2f;
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        //Using detection classifier
        if (mFaceDetector != null) {
            mFaceDetector.detectMultiScale(mGray, faces, 1.1, 5, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else {
            Log.e(TAG, "Detection is not selected!");
        }

        //Drawing rectangle around detected face
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        }

        //If one face is detected, method predictAge is executed
        if (facesArray.length == 1) {
            String age = predictAge(mRgba, facesArray);
            Log.i(TAG, "Age is: " + age);

            //The result of age recognition
            for (Rect face : facesArray) {
                int posX = (int) Math.max(face.tl().x - 10, 0);
                int posY = (int) Math.max(face.tl().y - 10, 0);

                Imgproc.putText(mRgba, "Godine: " + age, new Point(posX, posY), Core.FONT_HERSHEY_TRIPLEX,
                        1.5, new Scalar(0, 255, 0, 255));
            }
        }
        return mRgba;
    }

    //Method for age recognition
    private String predictAge(Mat mRgba, Rect[] facesArray) {
        try {
            for (Rect face : facesArray) {
                Mat capturedFace = new Mat(mRgba, face);
                //Resizing pictures to resolution of Caffe model
                Imgproc.resize(capturedFace, capturedFace, new Size(227, 227));
                //Converting RGBA to BGR
                Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

                //Forwarding picture through Dnn
                Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f, new Size(227, 227),
                        new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
                mAgeNet.setInput(inputBlob, "data");
                Mat probs = mAgeNet.forward("prob").reshape(1, 1);
                Core.MinMaxLocResult mm = Core.minMaxLoc(probs); //Getting largest softmax output

                double result = mm.maxLoc.x; //Result of age recognition prediction
                Log.i(TAG, "Result is: " + result);
                return AGES[(int) result];
            }
        } catch (Exception e) {
            Log.e(TAG, "Error processing age", e);
        }
        return null;
    }

    //Loading data from assets
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;

        try {
            //Reading data from app/build/intermediates/assets/debug
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            File outputFile = new File(context.getFilesDir(), file);
            FileOutputStream fileOutputStream = new FileOutputStream(outputFile);
            fileOutputStream.write(data);
            fileOutputStream.close();
            return outputFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }
}
