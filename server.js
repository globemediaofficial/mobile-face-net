import express from "express";
import fs from "fs";
import { Interpreter } from "node-tflite";

const app = express();
app.use(express.json({ limit: "20mb" }));

// Load MobileFaceNet model
const modelData = fs.readFileSync("./mobilefacenet.tflite");
const interpreter = new Interpreter(modelData);
interpreter.allocateTensors();
console.log("MobileFaceNet TFLite model loaded.");

app.post("/verifyFace", (req, res) => {
  try {
    const { images } = req.body;
    if (!images || images.length !== 2) 
      return res.status(400).json({ error: "Two images required" });

    const embeddings = images.map((base64) => {
      const buffer = Buffer.from(base64, "base64");
      const inputData = new Float32Array(buffer.buffer); // adjust type to match model

      // Copy input to tensor
      interpreter.inputs[0].copyFrom(inputData);

      // Run inference
      interpreter.invoke();

      // Get output embedding
      const outputData = new Float32Array(interpreter.outputs[0].size);
      interpreter.outputs[0].copyTo(outputData);

      return Array.from(outputData);
    });

    res.json(embeddings);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3291;
app.listen(PORT, () => console.log(`TFLite server running on port ${PORT}`));
