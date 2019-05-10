package sample;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import utils.Utils;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-10)
 * @since 1.0 (2014-01-10)
 *
 */
public class Controller
{
    // FXML buttons
    @FXML
    private Button cameraButton;
    // the FXML area for showing the current frame
    @FXML
    private ImageView originalFrame;
    // checkboxes for enabling/disabling a classifier
    @FXML
    private CheckBox haarClassifier;
    @FXML
    private CheckBox lbpClassifier;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that performs the video capture
    private VideoCapture capture;
    // a flag to change the button behavior
    private boolean cameraActive;



    // cascade classifiers
    private CascadeClassifier faceCascade;
    private CascadeClassifier eyeCascade;
    private CascadeClassifier mouthCascade;
    private CascadeClassifier noseCascade;
    private int absoluteFaceSize;

    //paths

    private String faceCascadePath = "resources/haarcascades/haarcascade_frontalface_alt_tree.xml";

    /**
     * Init the controller, at start time
     */
    protected void init()
    {
        this.capture = new VideoCapture();
        this.faceCascade = new CascadeClassifier();
        this.eyeCascade = new CascadeClassifier();
        this.mouthCascade = new CascadeClassifier();
        this.noseCascade = new CascadeClassifier();
        this.absoluteFaceSize = 0;

        // set a fixed width for the frame
        originalFrame.setFitWidth(600);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);
    }

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera()
    {
        if (!this.cameraActive)
        {
            // disable setting checkboxes
            this.haarClassifier.setDisable(true);
            this.lbpClassifier.setDisable(true);

            // start the video capture
            this.capture.open(0);

            // is the video stream available?
            if (this.capture.isOpened())
            {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run()
                    {
                        // effectively grab and process a single frame
                        Mat frame = grabFrame();
                        // convert and show the frame
                        Image imageToShow = Utils.mat2Image(frame);
                        updateImageView(originalFrame, imageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            }
            else
            {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        }
        else
        {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");
            // enable classifiers checkboxes
            this.haarClassifier.setDisable(false);
            this.lbpClassifier.setDisable(false);

            // stop the timer
            this.stopAcquisition();
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Mat grabFrame()
    {
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened())
        {
            try
            {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty())
                {
                    // face detection
                    this.detectAndDisplay(frame);
                }

            }
            catch (Exception e)
            {
                // log the (full) error
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return frame;
    }

    /**
     * Method for face detection and tracking
     *
     * @param frame
     *            it looks for faces in this frame
     */
    private void detectAndDisplay(Mat frame)
    {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();

        // convert the frame in gray scale
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (20% of the frame height, in our case)
        if (this.absoluteFaceSize == 0)
        {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0)
            {
                this.absoluteFaceSize = Math.round(height * 0.2f);
            }
        }

        // detect faces
        this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,  new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());

        // each rectangle in faces is a face: draw them!
        Rect[] facesArray = faces.toArray();
        MatOfRect eyes = new MatOfRect();
        this.eyeCascade.load("resources/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

        for (int i = 0; i < facesArray.length; i++) {
            Rect currentFace = facesArray[i];
            Imgproc.rectangle(frame, currentFace.tl(), currentFace.br(), new Scalar(0, 255, 0), 3);
            Imgproc.putText(frame, "face" + currentFace.tl().x + ", " + currentFace.tl().y, currentFace.tl(), 2, 2, new Scalar(0, 255, 0));

            Mat faceROI = grayFrame.submat(currentFace);



            this.eyeCascade.detectMultiScale(faceROI, eyes);
            Rect[] eyesArray = eyes.toArray();
            for (Rect eye : eyesArray) {
                Point tl = new Point(currentFace.x+eye.tl().x, currentFace.y+eye.tl().y);
                Point br = new Point(currentFace.x + eye.br().x, currentFace.y + eye.br().y);
                Imgproc.rectangle(frame, tl, br, new Scalar(0, 0, 255), 3);
                Imgproc.putText(frame, "Eye" + tl.x + ", " + tl.y, tl, 2, 2, new Scalar(0, 0, 255));
            }
        }









/**

        // mouth
        MatOfRect mouth = new MatOfRect();

        this.mouthCascade.load("resources/haarcascades/mouth.xml");
        // detect eyes
        this.mouthCascade.detectMultiScale(grayFrame, mouth, 1.1,4,0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize/5, this.absoluteFaceSize/5), new Size());
        Rect[] mouthArray = mouth.toArray();
        for (int i = 0; i < mouthArray.length; i++) {
            Imgproc.rectangle(frame, mouthArray[i].tl(), mouthArray[i].br(), new Scalar(255, 0, 0), 3);
            Imgproc.putText(frame, "mouth", mouthArray[i].tl(), 2, 2, new Scalar(255, 0, 0));
        }

        // nose
        MatOfRect nose = new MatOfRect();

        this.noseCascade.load("resources/haarcascades/nose.xml");
        // detect eyes
        this.noseCascade.detectMultiScale(grayFrame, nose, 1.1,2,0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize/5, this.absoluteFaceSize/5), new Size());
        Rect[] noseArray = eyes.toArray();
        for (int i = 0; i < noseArray.length; i++)
            Imgproc.rectangle(frame, noseArray[i].tl(), noseArray[i].br(), new Scalar(255, 255, 0), 3);

**/
    }

    /**
     * The action triggered by selecting the Haar Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void haarSelected(Event event)
    {
        // check whether the lpb checkbox is selected and deselect it
        if (this.lbpClassifier.isSelected())
            this.lbpClassifier.setSelected(false);

        this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");
    }

    /**
     * The action triggered by selecting the LBP Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void lbpSelected(Event event)
    {
        // check whether the haar checkbox is selected and deselect it
        if (this.haarClassifier.isSelected())
            this.haarClassifier.setSelected(false);

        this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
    }

    /**
     * Method for loading a classifier trained set from disk
     *
     * @param classifierPath
     *            the path on disk where a classifier trained set is located
     */
    private void checkboxSelection(String classifierPath)
    {
        // load the classifier(s)
        this.faceCascade.load(faceCascadePath);

        // now the video capture can start
        this.cameraButton.setDisable(false);
    }

    /**
     * Stop the acquisition from the camera and release all the resources
     */
    private void stopAcquisition()
    {
        if (this.timer!=null && !this.timer.isShutdown())
        {
            try
            {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            }
            catch (InterruptedException e)
            {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened())
        {
            // release the camera
            this.capture.release();
        }
    }

    /**
     * Update the {@link ImageView} in the JavaFX main thread
     *
     * @param view
     *            the {@link ImageView} to update
     * @param image
     *            the {@link Image} to show
     */
    private void updateImageView(ImageView view, Image image)
    {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed()
    {
        this.stopAcquisition();
    }

}