package com.kodnito.djl;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class DetectObject {

    private static final Logger logger = LoggerFactory.getLogger(DetectObject.class);

    public static void main(String[] args) throws MalformedModelException, ModelNotFoundException, TranslateException, IOException {
        var detectedObjects = new DetectObject().predict();
        logger.info("{}", detectedObjects);
    }

    public DetectedObjects predict() throws MalformedModelException, ModelNotFoundException, IOException, TranslateException {
        var imageFile = Paths.get("src/main/resources/new-york.jpg");
        var img = BufferedImageUtils.fromFile(imageFile);

        ZooModel<BufferedImage, DetectedObjects> model =
                MxModelZoo.SSD.loadModel(new ProgressBar());

        var predictor = model.newPredictor().predict(img);
        ImageVisualization.drawBoundingBoxes(img, predictor);
        ImageIO.write(img, "png", new File("new-york.png"));
        model.close();
        return predictor;
    }

}
