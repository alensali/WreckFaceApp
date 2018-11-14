package com.alensalihbasic.wreckfacejavacv;

import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ToggleButton;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
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
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static com.alensalihbasic.wreckfacejavacv.Methods.THRESHOLD;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class FaceRecognizerActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "FaceRecognizerActivity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier mFaceDetector;
    private File mCascadeFile;
    private Mat mRgba, mGray;
    private int mAbsoluteFaceSize = 0;
    private opencv_face.FaceRecognizer mLBPHFaceRecognizer = opencv_face.LBPHFaceRecognizer.create();
    private int mCameraId = 1;

    //Veza između aplikacije i OpenCV Manager-a
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
                                //Učitavanje datoteke klasifikatora iz resursa aplikacije
                                InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface_improved);
                                File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                                mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface_improved.xml");
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
        setContentView(R.layout.recognizer_face);
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
                Intent backIntent = new Intent(FaceRecognizerActivity.this, WelcomeScreen.class);
                startActivity(backIntent);
            }
        });

        //Učitavanje istreniranog klasifikatora iz direktorija SlikeLica u model za prepoznavanje lica
        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    File folder = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), Methods.FACE_PICS);
                    File f = new File(folder, Methods.LBPH_CLASSIFIER);
                    Log.i(TAG, "Classifier = " + f);
                    mLBPHFaceRecognizer.read(f.getAbsolutePath());
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }
        }.execute();

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

        //Izračunavanje absolutne veličine lica
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            float mRelativeFaceSize = 0.2f;
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        //Iskorištavanje klasifikatora za detekciju lica u slici
        if (mFaceDetector != null) {
            mFaceDetector.detectMultiScale(mGray, faces, 1.1, 5, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }else {
            Log.e(TAG, "Detection is not selected!");
        }


        //Crtanje pravokutnika oko svakog detektiranog lica u slici
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        }

        //Ako je detektirano jedno lice u slici, pokreće se metoda prepoznavanja lica
        if (facesArray.length == 1) {
            try {
                //Konverzija OpenCV Mat-a u JavaCV Mat
                opencv_core.Mat javaCvMat = new opencv_core.Mat((Pointer) null) {{address = mGray.getNativeObjAddr();}};
                //Smanjivanje rezolucije na 92x112
                resize(javaCvMat, javaCvMat, new opencv_core.Size(Methods.IMG_WIDTH, Methods.IMG_HEIGHT));
                //Ujednačavanje histograma
                equalizeHist(javaCvMat, javaCvMat);

                IntPointer label = new IntPointer(1);
                DoublePointer confidence = new DoublePointer(1);
                //Metoda prepoznavanja lica
                mLBPHFaceRecognizer.predict(javaCvMat, label, confidence);

                //Dohvaćanje najbliže slike iz treniranog modela lica
                int predictedLabel = label.get(0);
                //Dohvaćanje vjerojatnosti. Manji broj = preciznije.
                double acceptanceLevel = confidence.get(0);
                String name;
                Log.d(TAG, "Prediction completed, predictedLabel: " + predictedLabel + ", acceptanceLevel: " + acceptanceLevel);
                if (predictedLabel == -1 || acceptanceLevel >= THRESHOLD) {
                    name = "-";
                } else {
                    name = Integer.toString(predictedLabel);
                }

                //Ispis broja najbliže prepoznate slike lica iznad pravokutnika
                for (Rect face : facesArray) {
                    int posX = (int) Math.max(face.tl().x - 10, 0);
                    int posY = (int) Math.max(face.tl().y - 10, 0);
                    Imgproc.putText(mRgba, "Najbliza slika: br." + name, new Point(posX, posY),
                            Core.FONT_HERSHEY_TRIPLEX, 1.5, new Scalar(0, 255, 0, 255));
                }
            }catch (Exception e) {
                Log.d(TAG, e.getLocalizedMessage(), e);
            }
        }
        return mRgba;
    }
}
