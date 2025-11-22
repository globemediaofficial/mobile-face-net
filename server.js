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
console.log("MobileFaceNet TFLite model loaded.");

// Helper: preprocess image to Float32Array
async function preprocessImage(base64: string): Promise<Float32Array> {
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
    if (!images || images.length !== 2)
      return res.status(400).json({ error: "Two images required" });

    const embeddings = [];
    for (const base64 of images) {
      // Preprocess the image into Float32Array
      const inputData = await preprocessImage(base64);

      // Copy input to tensor
      interpreter.inputs[0].copyFrom(inputData);

      // Run inference
      interpreter.invoke();

      // Get output embedding
      const outputTensor = interpreter.outputs[0];
      const outputSize = outputTensor.shape.reduce((a, b) => a * b, 1);
      const outputData = new Float32Array(outputSize);
      outputTensor.copyTo(outputData);

      embeddings.push(Array.from(outputData));
    }

    res.json(embeddings);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3291;
app.listen(PORT, () =>
  console.log(`TFLite server running on port ${PORT}`)
);
