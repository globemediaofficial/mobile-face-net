import express from "express";
import fs from "fs";
import { Interpreter } from "node-tflite";

const app = express();
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Load model
const modelData = fs.readFileSync("./mobilefacenet.tflite");
const interpreter = new Interpreter(modelData);
interpreter.allocateTensors();

console.log("MobileFaceNet TFLite model loaded.");

// Example input processing (you may need to adjust for your model’s input)
app.post("/verifyFace", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: "No image provided" });

    const buffer = Buffer.from(image, "base64");
    // Convert buffer to Uint8Array
    const inputData = new Uint8Array(buffer);

    // Set input tensor: you might need shape/format adjustments
    interpreter.setInputTensorData(0, inputData);
    interpreter.invoke();
    const outputData = interpreter.getOutputTensorData(0);

    res.json({ embedding: Array.from(outputData) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3291;
app.listen(PORT, () => console.log(`TF Lite server running on port ${PORT}`));
