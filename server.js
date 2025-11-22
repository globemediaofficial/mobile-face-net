import express from "express";
import fs from "fs";
import sharp from "sharp";
import { Interpreter } from "node-tflite";

const app = express();
app.use(express.json({ limit: "50mb" }));

// Load MobileFaceNet model
const modelData = fs.readFileSync("./mobilefacenet.tflite");
const interpreter = new Interpreter(modelData);
interpreter.allocateTensors();
console.log("✅ MobileFaceNet TFLite model loaded.");

// Helper: preprocess image to Float32Array
async function preprocessImage(base64: string) {
  const buffer = Buffer.from(base64, "base64");

  // Resize to 112x112 and get raw RGB
  const raw = await sharp(buffer)
    .resize(112, 112)
    .removeAlpha()
    .raw()
    .toBuffer();

  const floatArray = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    floatArray[i] = raw[i] / 127.5 - 1; // normalize to [-1,1]
  }
  return floatArray;
}

app.post("/verifyFace", async (req, res) => {
  try {
    const { images } = req.body;
    if (!images || images.length !== 2) {
      console.warn("⚠️ Request body missing or incorrect images array:", images);
      return res.status(400).json({ error: "Two images required" });
    }

    console.log("📸 Received 2 images for verification");

    const embeddings = [];
    for (let idx = 0; idx < images.length; idx++) {
      const base64 = images[idx];

      // Preprocess the image into Float32Array
      const inputData = await preprocessImage(base64);

      // Log input tensor info
      const inputTensor = interpreter.inputs[0];
      if (!inputTensor) {
        console.error("❌ Input tensor is undefined");
        return res.status(500).json({ error: "Input tensor undefined" });
      }
      console.log(`🟢 Image ${idx} input tensor shape:`, inputTensor.shape);

      // Copy input to tensor
      try {
        inputTensor.copyFrom(inputData);
      } catch (err) {
        console.error("❌ Failed to copy input data to tensor:", err);
        return res.status(500).json({ error: "Failed to copy input data to tensor" });
      }

      // Run inference
      try {
        interpreter.invoke();
      } catch (err) {
        console.error("❌ Interpreter failed during invoke:", err);
        return res.status(500).json({ error: "Interpreter invoke failed" });
      }

      // Get output embedding
      const outputTensor = interpreter.outputs[0];
      if (!outputTensor) {
        console.error("❌ Output tensor is undefined");
        return res.status(500).json({ error: "Output tensor undefined" });
      }
      if (!outputTensor.shape) {
        console.error("❌ Output tensor shape is undefined");
        return res.status(500).json({ error: "Output tensor shape undefined" });
      }

      console.log(`🟢 Image ${idx} output tensor shape:`, outputTensor.shape);

      // Copy output data
      const outputSize = outputTensor.shape.reduce((a, b) => a * b, 1);
      const outputData = new Float32Array(outputSize);
      try {
        outputTensor.copyTo(outputData);
      } catch (err) {
        console.error("❌ Failed to copy output data:", err);
        return res.status(500).json({ error: "Failed to copy output data" });
      }

      embeddings.push(Array.from(outputData));
    }

    console.log("✅ Successfully generated embeddings for both images");
    res.json(embeddings);
  } catch (err) {
    console.error("❌ Unexpected error in /verifyFace:", err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3291;
app.listen(PORT, () => console.log(`🚀 TFLite server running on port ${PORT}`));
