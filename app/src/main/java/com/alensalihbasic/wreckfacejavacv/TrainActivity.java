package com.alensalihbasic.wreckfacejavacv;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class TrainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "TrainActivity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier mFaceDetector;
    private File mCascadeFile;
    private Mat mRgba, mGray;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;
    private boolean takePhoto, trained;
    private String personName;

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
                                // load cascade file from application resources
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
                                }else {
                                    Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                                }
                                cascadeDir.delete();

                            }catch (IOException e) {
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
        }else {
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
        setContentView(R.layout.activity_train);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        SharedPreferences myPrefs = getApplicationContext().getSharedPreferences("myPrefs", MODE_PRIVATE);
        personName = myPrefs.getString("name", null);
        Log.v(TAG, "mName is " + personName);

        findViewById(R.id.take_pic_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                takePhoto = true;
            }
        });

        findViewById(R.id.train_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!trained) {
                    train();
                }
            }
        });

        findViewById(R.id.reset_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    Methods.reset(getBaseContext());
                    trained = false;
                    Toast.makeText(TrainActivity.this, "Uspješno poništavanje", Toast.LENGTH_SHORT).show();
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
            }
        });

        findViewById(R.id.recognize_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Methods.isTrained(getBaseContext())) {
                    Intent faceRecognizerActivityIntent = new Intent(TrainActivity.this, FaceRecognizerActivity.class);
                    startActivity(faceRecognizerActivityIntent);
                }else {
                    Toast.makeText(TrainActivity.this, "Morate prvo istrenirati", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
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

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();
        // Use the classifier to detect faces
        if (mFaceDetector != null) {
            mFaceDetector.detectMultiScale(mGray, faces, 1.1, 5, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }else {
            Log.e(TAG, "Detection is not selected!");
        }

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        }

        if (facesArray.length == 1) {
            if (takePhoto) {
                capturePhoto(mRgba);
                alertRemainingPhotos();
            }
        }

        return mRgba;
    }

    private void capturePhoto(Mat rgbaMat) {
        try {
            Methods.takePhoto(getBaseContext(), personName, Methods.qtdPhotos(getBaseContext()) + 1, rgbaMat.clone(), mFaceDetector);
        }catch (Exception e) {
            e.printStackTrace();
        }
        takePhoto = false;
    }

    private void train() {
        int remainingPhotos = Methods.PHOTOS_TRAIN_QTY - Methods.qtdPhotos(getBaseContext());
        if (remainingPhotos > 0) {
            Toast.makeText(this, "Trebate još " + remainingPhotos + " sliku/a za treniranje", Toast.LENGTH_SHORT).show();
            return;
        }else if (Methods.isTrained(getBaseContext())) {
            Toast.makeText(this, "Istrenirano već", Toast.LENGTH_SHORT).show();
            return;
        }

        Toast.makeText(this, "Treniranje započeto", Toast.LENGTH_SHORT).show();
        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    if (!Methods.isTrained(getBaseContext())) {
                        Methods.train(getBaseContext());
                        trained = true;
                    }
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                try {
                    Toast.makeText(TrainActivity.this, "Uspjeh treniranja: " + Methods.isTrained(getBaseContext()), Toast.LENGTH_SHORT).show();
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
            }
        }.execute();
    }

    private void alertRemainingPhotos() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                int remainingPhotos = Methods.PHOTOS_TRAIN_QTY - Methods.qtdPhotos(getBaseContext());
                if (remainingPhotos > 0) {
                    Toast.makeText(getBaseContext(), "Još " + remainingPhotos + " slika", Toast.LENGTH_SHORT).show();
                }else {
                    Toast.makeText(TrainActivity.this, "Slikali ste maksimum broj slika", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}
