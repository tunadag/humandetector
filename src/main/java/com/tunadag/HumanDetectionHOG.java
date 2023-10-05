package com.tunadag;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

public class HumanDetectionHOG {

    public static void main(String[] args) {
        // Load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Initialize the HOG descriptor/person detector
        HOGDescriptor hog = new HOGDescriptor();
        hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());

        // Open webcam video stream
        VideoCapture cap = new VideoCapture(0);

        // Create a Mat to store the frame
        Mat frame = new Mat();

        // Specify the directory where images will be saved
        String saveDir = "C:/Users/tuna_/Desktop/insandeneme/";

        while (true) {
            // Capture frame-by-frame
            cap.read(frame);

            // Resize frame for faster detection
            Imgproc.resize(frame, frame, new Size(640, 480));

            // Convert the frame to grayscale for faster detection
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY);

            // Detect people in the image
            MatOfRect found = new MatOfRect();
            MatOfDouble weights = new MatOfDouble();
            hog.detectMultiScale(frame, found, weights, 0, new Size(8, 8),
                    new Size(16, 16), 1.03, 2.0, false);

            Rect[] boxes = found.toArray();

            // Check if any humans are detected
            if (boxes.length > 0) {
                // Save the entire frame
                String filename = saveDir + "frame_" + System.currentTimeMillis() + ".jpg";
                Imgcodecs.imwrite(filename, frame);
            }

            for (Rect box : boxes) {
                // Display the detected boxes in the color picture
                Imgproc.rectangle(frame, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height), new Scalar(0, 255, 0), 2);
            }

            // Display the resulting frame
            HighGui.imshow("frame", frame);

            // Check for the 'esc' key press to exit
            int key = HighGui.waitKey(1) & 0xFF;
            if (key == 27) { // 27, 'esc' tuşunun ASCII kodu
                cap.release();
                HighGui.destroyAllWindows();
                System.exit(0); // Programı kapat
            }
        }
    }
}
