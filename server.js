import express from "express";
import * as tflite from "@tensorflow/tfjs-tflite";
import fs from "fs";
import * as tf from "@tensorflow/tfjs-node";

const app = express();
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Load TFLite model
const modelBuffer = fs.readFileSync("./mobilefacenet.tflite");
const model = new tflite.TFLiteModel(modelBuffer);

console.log("MobileFaceNet TFLite model loaded.");

// Endpoint for verifying face
app.post("/verifyFace", async (req, res) => {
  try {
    const { image } = req.body; // image as base64
    if (!image) return res.status(400).json({ error: "No image provided" });

    const buffer = Buffer.from(image, "base64");

    // Preprocess image: decode -> resize -> normalize
    let tensor = tf.node.decodeImage(buffer, 3)
                  .resizeNearestNeighbor([112, 112])
                  .expandDims(0)
                  .toFloat()
                  .div(tf.scalar(255.0));

    // Run inference
    const output = await model.predict(tensor);

    res.json({ embedding: Array.from(output.dataSync()) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3291;
app.listen(PORT, () => console.log(`TF Lite server running on port ${PORT}`));
