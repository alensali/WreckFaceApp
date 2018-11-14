package com.alensalihbasic.wreckfacejavacv;

import android.os.Environment;
import android.util.Log;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_imgcodecs;
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
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.opencv.core.CvType.CV_32SC1;

public class Methods {
    public static final String TAG = "Methods";
    public static final String FACE_PICS = "SlikeLica";
    public static final int IMG_WIDTH = 92;
    public static final int IMG_HEIGHT = 112;
    public static final int PHOTOS_TRAIN_QTY = 5;
    public static final double THRESHOLD = 125.0D;
    public static final String LBPH_CLASSIFIER = "lbphKlasifikator.xml";
    public static final File ROOT = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), FACE_PICS);

    //Metoda brisanja svih datoteka iz direktorija SlikeLica
    public static void reset() {
        File facePicsPath = new File(String.valueOf(ROOT));

        if (facePicsPath.exists()) {
            File[] facePicsArray = facePicsPath.listFiles();
            for (File facepicsTmp : facePicsArray) {
                facepicsTmp.delete();
            }
        }
    }

    //Metoda koja provjerava da li su ispunjeni uvjeti: broj slika i istrenirani model
    public static boolean isTrained() {
        try {
            File facePicsPath = new File(String.valueOf(ROOT));

            if (facePicsPath.exists()) {
                FilenameFilter photoFilter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.endsWith(".png");
                    }
                };
                FilenameFilter trainFilter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.endsWith(".xml");
                    }
                };
                File[] photosArray = facePicsPath.listFiles(photoFilter);
                File[] train = facePicsPath.listFiles(trainFilter);
                return photosArray != null && train != null && photosArray.length == PHOTOS_TRAIN_QTY && train.length > 0;
            } else {
                return false;
            }
        } catch (Exception e) {
            Log.d(TAG, e.getLocalizedMessage(), e);
        }
        return false;
    }

    //Metoda provjere broja slika lica
    public static int numPhotos() {
        File facePicsPath = new File(String.valueOf(ROOT));

        if (facePicsPath.exists()) {
            FilenameFilter photoFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith(".png");
                }
            };
            File[] numPhotoArray = facePicsPath.listFiles(photoFilter);
            return numPhotoArray != null ? numPhotoArray.length : 0;
        }
        return 0;
    }

    //Metoda treniranja modela prepoznavanja lica
    public static boolean train() throws Exception {
        File facePicsPath = new File(String.valueOf(ROOT));

        if (!facePicsPath.exists()) {
            return false;
        }
        FilenameFilter photoFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".png");
            }
        };
        File[] photosArray = facePicsPath.listFiles(photoFilter);
        opencv_core.MatVector photosMatVector = new opencv_core.MatVector(photosArray.length);
        opencv_core.Mat labels = new opencv_core.Mat(photosArray.length, 1, CV_32SC1);
        IntBuffer intBuffer = labels.createBuffer();
        int counter = 0;

        for (File image : photosArray) {
            //Učitavanje slike u jednokanalnom formatu slike (GRAY)
            opencv_core.Mat photo = imread(image.getAbsolutePath(), opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Log.d(TAG, "Image: " + image.getName());
            //Odvajanje broja slike
            int intLabel = Integer.parseInt(image.getName().split("\\.")[0]);
            //Smanjivanje rezolucije na 92x112
            resize(photo, photo, new opencv_core.Size(IMG_WIDTH, IMG_HEIGHT));
            //Ujednačavanje histograma slike
            equalizeHist(photo, photo);
            photosMatVector.put(counter, photo);
            intBuffer.put(counter, intLabel);
            counter++;
        }

        //Kreiranje, treniranje i zapisivanje LBPH modela prepoznavanja lica
        opencv_face.FaceRecognizer mLBPHFaceRecognizer = opencv_face.LBPHFaceRecognizer.create();
        mLBPHFaceRecognizer.train(photosMatVector, labels);
        File trainedFaceRecognizerModel = new File(facePicsPath, LBPH_CLASSIFIER);
        trainedFaceRecognizerModel.createNewFile();
        mLBPHFaceRecognizer.write(trainedFaceRecognizerModel.getAbsolutePath());
        return true;
    }

    //Metoda uzimanja i spremanja slika
    public static void takePhoto(int photoNumber, Mat rgbaMat, CascadeClassifier faceDetector) throws Exception {
        File facePicsPath = new File(String.valueOf(ROOT));

        if (facePicsPath.exists() && !facePicsPath.isDirectory())
            facePicsPath.delete();
        if (!facePicsPath.exists())
            facePicsPath.mkdirs();

        Mat grayMat = new Mat();
        //Pretvaranje četverokanalne slike (RGBA) u jednokanalnu sliku (GRAY)
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);

        MatOfRect detectedFaces = new MatOfRect();
        faceDetector.detectMultiScale(grayMat, detectedFaces);
        Rect[] detectedFacesArray = detectedFaces.toArray();

        for (Rect face : detectedFacesArray) {
            Mat capturedFace = new Mat(grayMat, face);
            //Smanjivanje rezolucije na 92x112
            Imgproc.resize(capturedFace, capturedFace, new Size(IMG_WIDTH, IMG_HEIGHT));
            //Ujednačavanje histograma slike
            Imgproc.equalizeHist(capturedFace, capturedFace);

            if (photoNumber <= PHOTOS_TRAIN_QTY) {
                //Format slike u obliku brojSlike.formatSlike (pr. 1.png)
                File savePhoto = new File(facePicsPath, String.format("%d.png", photoNumber));
                savePhoto.createNewFile();
                //Zapisivanje slike lica u direktorij SlikeLica
                Imgcodecs.imwrite(savePhoto.getAbsolutePath(), capturedFace);
                Log.i(TAG, "PIC PATH: " + savePhoto.getAbsolutePath());
            }
        }
    }
}
