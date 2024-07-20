# imports 
import os  # For file operations
import cv2  # For image processing
import xml.etree.ElementTree as ET  # For parsing XML files
import numpy as np
import pandas as pd
import easyocr  # For text extraction from license plates
from deployment_udp_client import stream_video  # For streaming video from a given input URL using ffmpeg and displaying it with OpenCV
# from tqdm import tqdm  # For progress bar

# define Data Pipeline class
class DataPipeline:
    def __init__(self):
        self.data = None

        # Image dimensions, for resizing the input images
        self.IMAGE_WIDTH = 416
        self.IMAGE_HEIGHT = 416

        # For image preprocessing
        self.cropped_images = []
        self.preprocessed_images = None

        # For object detection with YOLOv3
        self.model = None
        self.model_tiny = None  # For faster inference
        self.bounding_boxes = None

        # For text extraction with EasyOCR
        self.results = None

    def normalize(self, image):
        # Normalize the input image for OCR
        norm_image = np.zeros((image.shape[0], image.shape[1]))
        img = cv2.normalize(image, norm_image, 0, 255, cv2.NORM_MINMAX)
        return img
    
    def resize(self, image):
        # Resize the input image for OCR
        resized_image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
        return resized_image
    
    def remove_noise(self, image):
        # Remove noise from the input image for OCR
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        return denoised_image

    def skeletonize(self, image):
        kernel = np.ones((5,5),np.uint8)
        skeletonized_image = cv2.erode(image, kernel, iterations = 1)
        return skeletonized_image

    def sharpen(self, image):
        # Sharpen the input image for OCR
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image
    
    def grayscale(self, image):
        # Convert the input image to grayscale for OCR
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def thresholding(self, image):
        # Apply thresholding to the input image for OCR
        ret, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_image
    
    def preprocess(self, image):
        # Placeholder for the culmination of all image preprocessing steps to perform on the cropped images
        # Preprocessing steps to prepare images for OCR
        
        # Normalize
        image = self.normalize(image)
        # Resize
        image = self.resize(image)
        # Remove noise
        image = self.remove_noise(image)
        # Skeletonize
        image = self.skeletonize(image)
        # Sharpen
        image = self.sharpen(image)
        # Convert to grayscale
        image = self.grayscale(image)
        # Apply thresholding
        image = self.thresholding(image)

        return image

    def load_object_detection_model(self, tiny=False):
        # Load the object detection model
        if tiny:
            model_weights = "lpr-yolov3-tiny.weights"
            config = "lpr-yolov3-tiny.cfg"
            self.model_tiny = cv2.dnn.readNet(model_weights, config)
        else:
            model_weights = "lpr-yolov3.weights"
            config = "lpr-yolov3.cfg"
            self.model = cv2.dnn.readNet(model_weights, config)
    
    # test object detection on a toy imageset first, to make sure it works
    def object_detection(self, image, conf_threshold=0.5, nms_threshold=0.4, tiny=False):
        # Perform object detection on the input image using YOLOv3
        if tiny:
            model = self.model_tiny
        else:
            model = self.model

        # Get the output layer names
        layer_names = model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

        # Create a blob from the input image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), (0, 0, 0), True, crop=False)

        # Perform a forward pass of the YOLO object detector
        model.setInput(blob)
        outs = model.forward(output_layers)

        # Get the bounding boxes, confidences, and class IDs
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    # Object detected
                    center_x = int(detection[0] * 3840)
                    center_y = int(detection[1] * 2160)
                    w = int(detection[2] * 3840)
                    h = int(detection[3] * 2160)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        bounding_boxes = []
        # Draw the bounding boxes on the input image
        for i in indices:
            box = boxes[i]
            bounding_boxes.append(box)  # Only save the most accurate bounding boxes
            # x = box[0]
            # y = box[1]
            # w = box[2]
            # h = box[3]
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.bounding_boxes = bounding_boxes
        return image
    
    def extract(self, video):
        # Stream video from a given input URL using ffmpeg and display it with OpenCV
        stream_video(video, 3840, 2160)

        # Take input video (path) and extract frames
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        count = 0

        # Check if the directory "frames" already exists
        if not os.path.exists("frames") or not os.listdir("frames"):
            # Create a directory to store the frames
            os.makedirs("frames", exist_ok=True)
        else:
            print("Directory 'frames' already exists and is not empty.")
            frames = []
            for file in os.listdir("frames"):
                frames.append(cv2.imread("frames/" + file))
            self.data = frames
            return

        # frames = []
        # total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # with tqdm(total=total_frames) as pbar:
        #     while success:
        #         if count % 60 == 0:
        #             cv2.imwrite("frames/frame%d.jpg" % count, image) # save frame as JPEG file
        #             frames.append(image)
        #             pbar.update(1)
        #         count += 1 
        #         success,image = vidcap.read()

        frames = []
        while success:
            if count % 60 == 0:
                cv2.imwrite("frames/frame%d.jpg" % count, image) # save frame as JPEG file
                frames.append(image)
                print(count)
            count += 1
            success,image = vidcap.read()
        
        self.data = frames
        print("Image data extracted successfully from video.")

    def transform(self, tiny_model=False):
        # Load object detection model
        self.load_object_detection_model(tiny=tiny_model)
        print("Object detection model loaded successfully.")
        
        # Crop the license plates from the input images
        cropped_images = []
        for image in self.data:
            self.object_detection(image)
            for box in self.bounding_boxes:
                x, y, w, h = box
                cropped_image = image[y:y+h, x:x+w]
                cropped_images.append(cropped_image)

        self.cropped_images = cropped_images
        print("Images cropped successfully.")

        # Preprocess the cropped images
        preprocessed_images = []
        for cropped_image in self.cropped_images:
            preprocessed_image = self.preprocess(cropped_image)
            preprocessed_images.append(preprocessed_image)
        self.preprocessed_images = preprocessed_images
        print("Image preprocessing steps completed successfully.")
    
    def load(self):
        # Loading the preprocessed images into EasyOCR for text extraction
        reader = easyocr.Reader(['en'])
        results = []

        # Initialize the OCR reader
        for image in self.preprocessed_images:
            result = reader.readtext(image)
            results.append(result)
        print("Text extraction completed successfully.")
        
        # Convert EasyOCR results to pandas DataFrame
        data = []
        for result in results:
            for detection in result:
                text = detection[1]
                confidence = detection[2]
                data.append({'Text': text, 'Confidence': confidence})

        df = pd.DataFrame(data)
        df.to_csv("ocr_results.csv", index=False)  # Save the OCR results to a CSV file
        self.results = df

        print("OCR Results saved successfully.")
        return df