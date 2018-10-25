package com.alensalihbasic.wreckfacejavacv;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class Methods {
    public static final String TAG = "Methods";
    public static final String FACE_PICS = "FacePics";
    public static final String TRAIN_FOLDER = "TrainFolder";
    public static final int IMG_WIDTH = 92;
    public static final int IMG_HEIGHT = 112;
    public static final int PHOTOS_TRAIN_QTY = 25;
    public static final double ACCEPT_LEVEL = 4000.0D;
    public static final String LBPH_CLASSIFIER = "lbphClassifier.xml";

    public static void reset(Context context) throws Exception {
        File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File path = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), FACE_PICS);
        if (path.exists()) {
            File[] pics = path.listFiles();
            for (File tmp : pics) {
                tmp.delete();
            }
        }
        if(photosFolder.exists()) {

            FilenameFilter imageFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return  name.endsWith(".png")  || name.endsWith(".xml");
                }
            };

            File[] files = photosFolder.listFiles(imageFilter);
            for(File file : files) {
                file.delete();
            }
        }
    }

    public static boolean isTrained(Context context) {
        try {
            File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
            if(photosFolder.exists()) {

                FilenameFilter imageFilter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return  name.endsWith(".png");
                    }
                };

                FilenameFilter trainFilter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return  name.endsWith(".xml");
                    }
                };

                File[] photos = photosFolder.listFiles(imageFilter);
                File[] train = photosFolder.listFiles(trainFilter);
                return photos!= null && train!= null && photos.length == PHOTOS_TRAIN_QTY && train.length > 0;
            } else {
                return false;
            }

        }catch (Exception e) {
            Log.d(TAG, e.getLocalizedMessage(), e);
        }
        return false;
    }

    public static int qtdPhotos(Context context) {
        File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        if(photosFolder.exists()) {
            FilenameFilter imageFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return  name.endsWith(".png");
                }
            };

            File[] files = photosFolder.listFiles(imageFilter);
            return files != null ? files.length : 0;
        }
        return 0;
    }

    public static boolean train(Context context) throws Exception{
        File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        if(!photosFolder.exists()) return false;

        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return  name.endsWith(".png");
            }
        };

        File[] files = photosFolder.listFiles(imageFilter);
        opencv_core.MatVector photos = new opencv_core.MatVector(files.length);
        opencv_core.Mat labels = new opencv_core.Mat(files.length, 1, CV_32SC1);
        IntBuffer intBuffer = labels.createBuffer();
        int counter = 0;
        for (File image : files) {
            opencv_core.Mat photo = imread(image.getAbsolutePath(), opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            int label = Integer.parseInt(image.getName().split("\\.")[1]);
            resize(photo, photo, new opencv_core.Size(IMG_WIDTH, IMG_HEIGHT));
            equalizeHist(photo, photo);
            photos.put(counter, photo);
            intBuffer.put(counter, label);
            counter++;
        }

        opencv_face.FaceRecognizer eigenFaceRecognizer = opencv_face.EigenFaceRecognizer.create();
        eigenFaceRecognizer.train(photos, labels);
        File f = new File(photosFolder, LBPH_CLASSIFIER);
        f.createNewFile();
        eigenFaceRecognizer.save(f.getAbsolutePath());

        return true;
    }

    public static void takePhoto(Context context, String personName, int photoNumber, Mat rgbaMat, CascadeClassifier faceDetector) throws Exception {
        File folder = new File(context.getFilesDir(), TRAIN_FOLDER);
        if (folder.exists() && !folder.isDirectory())
            folder.delete();
        if (!folder.exists())
            folder.mkdirs();

        Mat greyMat = new Mat();
        Imgproc.cvtColor(rgbaMat, greyMat, Imgproc.COLOR_BGR2GRAY);

        MatOfRect detectedFaces = new MatOfRect();
        faceDetector.detectMultiScale(greyMat, detectedFaces);
        Rect[] detectedFacesArray = detectedFaces.toArray();
        for (Rect face : detectedFacesArray) {
            Mat capturedFace = new Mat(greyMat, face);
            Imgproc.resize(capturedFace, capturedFace, new Size(IMG_WIDTH, IMG_HEIGHT));
            Imgproc.equalizeHist(capturedFace, capturedFace);

            if (photoNumber <= PHOTOS_TRAIN_QTY) {
                File f = new File(folder, String.format("%s.%d.png", personName, photoNumber));
                f.createNewFile();
                Imgcodecs.imwrite(f.getAbsolutePath(), capturedFace);
                Log.i(TAG, "PIC PATH: " + f.getAbsolutePath());
                SavePhoto(capturedFace, personName, photoNumber);
                Log.i(TAG, "Success in taking photo");
            }
        }
    }

    private static void SavePhoto(Mat rgbaMat, String personName, int photoNumber) {
        File path = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), FACE_PICS);
        path.mkdir();
        String filename = String.format("%s.%d.png", personName, photoNumber);
        File file = new File(path, filename);

        Boolean bool = imwrite(file.toString(), rgbaMat);

        if (bool) {
            Log.i(TAG, "SUCCESS writing image");
        }else {
            Log.i(TAG, "FAILED writing image");
        }
    }

}
