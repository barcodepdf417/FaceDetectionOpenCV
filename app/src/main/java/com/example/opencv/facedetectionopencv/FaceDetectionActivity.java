package com.example.opencv.facedetectionopencv;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.View;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.example.opencv.facedetectionopencv.pojo.Eyes;
import com.example.opencv.facedetectionopencv.pojo.Face;
import com.google.gson.Gson;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


public class FaceDetectionActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;

    private int learnFrames = 0;
    private Mat teplateR;
    private Mat teplateL;
    int method = 0;
    int smiled = 0;

    private Mat ocvRgb;
    private Mat ocvGray;

    private CascadeClassifier javaDetector;
    private CascadeClassifier javaDetectorEye;
    private CascadeClassifier javaDetectorSmile;

    private String[] detectorName;

    private float relativeFaceSize = 0.2f;
    private int absoluteFaceSize = 0;

    private CameraBridgeViewBase openCvCameraView;

    private SeekBar mMethodSeekbar;

    private Rect prevRightEye;
    private Rect prevLeftEye;

    private TextView blinksView;
    private TextView smileView;
    private volatile int blinks;
    private boolean isChanged = false;

    double xCenter = -1;
    double yCenter = -1;

    private List<Face> facesList;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        javaDetector = loadCascade(R.raw.lbpcascade_frontalface, "lbpcascade_frontalface.xml");
                        javaDetectorEye = loadCascade(R.raw.haarcascade_lefteye_2splits, "haarcascade_eye_right.xml");
                        javaDetectorSmile = loadCascade(R.raw.haarcascade_smile, "haarcascade_smile.xml");
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    openCvCameraView.setCameraIndex(1);
                    openCvCameraView.enableFpsMeter();
                    openCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    static{
        OpenCVLoader.initDebug();
        System.loadLibrary("opencv_java");
    }

    public FaceDetectionActivity() {
        detectorName = new String[2];
        detectorName[JAVA_DETECTOR] = "Java";
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_face_detection);

        facesList = new ArrayList<Face>();
        openCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        openCvCameraView.setCvCameraViewListener(this);

        mMethodSeekbar = (SeekBar) findViewById(R.id.methodSeekBar);
        blinksView = (TextView) findViewById(R.id.blinks);
        smileView = (TextView) findViewById(R.id.smiles);

        mMethodSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub

            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub

            }

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                method = progress;
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (openCvCameraView != null)
            openCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        openCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        ocvGray = new Mat();
        ocvRgb = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        ocvGray.release();
        ocvRgb.release();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        return true;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        ocvRgb = inputFrame.rgba();
        ocvGray = inputFrame.gray();

        if (absoluteFaceSize == 0) {
            int height = ocvGray.rows();
            if (Math.round(height * relativeFaceSize) > 0) {
                absoluteFaceSize = Math.round(height * relativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();
        if (javaDetector != null)
            javaDetector.detectMultiScale(ocvGray, faces, 1.1, 2, 2, // TODO:
                    // objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());

        Rect[] facesArray = faces.toArray();
        for (Rect face : facesArray) {
            Face jsonFace = new Face();
            jsonFace.setFaceId(1L);
            jsonFace.setTime(System.currentTimeMillis());
            Eyes jsonEyes = new Eyes();
            detectSmile(face, jsonFace);

            Core.rectangle(ocvRgb, face.tl(), face.br(), FACE_RECT_COLOR, 3);
            xCenter = (face.x + face.width + face.x) / 2;
            yCenter = (face.y + face.y + face.height) / 2;
            Point center = new Point(xCenter, yCenter);

//            Core.circle(ocvRgb, center, 10, new Scalar(255, 0, 0, 255), 3);

//            Core.putText(ocvRgb, "[" + center.x + "," + center.y + "]", new Point(center.x + 20, center.y + 20), Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255,
//                    255, 255, 255));

            Rect r = face;
            // compute the eye area
            Rect eyearea = new Rect(r.x + r.width / 8, (int) (r.y + (r.height / 4.5)), r.width - 2 * r.width / 8, (int) (r.height / 3.0));
            // split it
            Rect eyeareaRight = new Rect(r.x + r.width / 16, (int) (r.y + (r.height / 4.5)), (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            Rect eyeareaLeft = new Rect(r.x + r.width / 16 + (r.width - 2 * r.width / 16) / 2, (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            // draw the area - mGray is working grayscale mat, if you want to
            // see area in rgb preview, change mGray to mRgba
            Core.rectangle(ocvRgb, eyeareaLeft.tl(), eyeareaLeft.br(), new Scalar(255, 0, 0, 255), 2);
            Core.rectangle(ocvRgb, eyeareaRight.tl(), eyeareaRight.br(), new Scalar(255, 0, 0, 255), 2);

            if (learnFrames < 5) {
                teplateR = getTemplate(javaDetectorEye, eyeareaRight, 5);
                teplateL = getTemplate(javaDetectorEye, eyeareaLeft, 5);
                learnFrames++;
                jsonEyes.setLeftClosed(false);
                jsonEyes.setRightClosed(false);
            } else {
                // Learning finished, use the new templates for template
                // matching
                Rect currentRightRect = matchEye(eyeareaRight, teplateR, method);
                Rect currentLeftRect = matchEye(eyeareaLeft, teplateL, method);
                boolean rightBlinked = false;
                boolean leftBlinked = false;

                if (currentLeftRect != null && currentLeftRect.height > 0) {
                    leftBlinked = isBlinked(currentLeftRect, prevLeftEye);
                    prevLeftEye = currentLeftRect;
                    jsonEyes.setLeftClosed(leftBlinked);
                }
                if (currentRightRect != null && currentRightRect.height > 0) {
                    rightBlinked = isBlinked(currentRightRect, prevRightEye);
                    prevRightEye = currentRightRect;
                    jsonEyes.setRightClosed(rightBlinked);
                }

                if (rightBlinked || leftBlinked) {
                    ++blinks;
                    this.runOnUiThread(new Runnable() {

                        @Override
                        public void run() {
                            blinksView.setText(Integer.toString(blinks));
                        }
                    });
                }
            }
            jsonFace.setEyes(jsonEyes);
            facesList.add(jsonFace);
        }
        return ocvRgb;
    }

    private ArrayList<Mat> detectSmile(Rect face, Face jsonFace) {
        ArrayList<Mat> mouths = new ArrayList<Mat>();
        MatOfRect mouth = new MatOfRect();

        if (javaDetectorSmile != null)
            javaDetectorSmile.detectMultiScale(ocvGray, mouth, 1.1, 2,
                    2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size());

        Rect[] mouthArray = mouth.toArray();
        for (int i = 0; i < mouthArray.length; i++) {
            if (mouthArray[i].y > face.y + face.height * 3 / 5 && mouthArray[i].y + mouthArray[i].height < face.y + face.height
                    && Math.abs((mouthArray[i].x + mouthArray[i].width / 2)) - (face.x + face.width / 2) < face.width / 10) {
                ++smiled;
                jsonFace.setSmile(true);
                Core.rectangle(ocvRgb, mouthArray[i].tl(), mouthArray[i].br(), new Scalar(0, 255, 0, 255), 2);

                this.runOnUiThread(new Runnable() {

                    @Override
                    public void run() {
                        smileView.setText("Smiles "  + Integer.toString(smiled));
                    }
                });
            } else {
                jsonFace.setSmile(false);
            }
        }
        return mouths;
    }

    private CascadeClassifier loadCascade(int rawId, String xmlFile) throws IOException {
        File cascadeFile;
        InputStream is = getResources().openRawResource(rawId);
        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
        cascadeFile = new File(cascadeDir, xmlFile);
        FileOutputStream os = new FileOutputStream(cascadeFile);

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        is.close();
        os.close();

        CascadeClassifier cascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
        if (cascade.empty()) {
            Log.e(TAG, "Failed to load cascade classifier");
            cascade = null;
        } else
            Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());

        cascadeFile.delete();
        return cascade;
    }

    private boolean isBlinked(Rect current, Rect previous) {
        if (previous != null && current != null) {
            int diffX = Math.abs(current.x - previous.x);
            int diffY = Math.abs(current.y - previous.y);
            Log.i("MYLOG", "Diff x=" + Integer.toString(diffX) + " y=" + Integer.toString(diffY));
            if (diffX > 30 || diffY > 30) {
                if (!isChanged) {
                    isChanged = true;
                    return false;
                } else {
                    isChanged = false;
                    return true;
                }
            }
        }
        return false;
    }

    private Rect matchEye(Rect area, Mat mTemplate, int type) {
        Point matchLoc;
        Mat mROI = ocvGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return null;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (type) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x, matchLoc.y + mTemplate.rows() + area.y);

        Core.rectangle(ocvRgb, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 255, 0));
        return new Rect(new Point(matchLoc.x, matchLoc.y), new Point(matchLoc.x + mTemplate.cols(), matchLoc.y + mTemplate.rows()));
    }

    private Mat getTemplate(CascadeClassifier clasificator, Rect area, int size) {
        Mat template = new Mat();
        Mat mROI = ocvGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(20, 20), new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4), e.width, (int) (e.height * 0.6));
            mROI = ocvGray.submat(eye_only_rectangle);
            Mat vyrez = ocvRgb.submat(eye_only_rectangle);

            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Core.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y - size / 2, size, size);
            Core.rectangle(ocvRgb, eye_template.tl(), eye_template.br(), new Scalar(255, 0, 0, 255), 2);
            template = (ocvGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    public void onRecreateClick(View v) {
        learnFrames = 0;
    }

    public void showJson(View view) {
        Gson gson = new Gson();
        String json = gson.toJson(facesList);
        Toast.makeText(this, json, Toast.LENGTH_SHORT).show();
    }
}
