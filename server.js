import express from "express";
import tflite from "tflite-node";
import fs from "fs";

const app = express();
app.use(express.json({ limit: "10mb" }));

// Load TFLite model
const model = new tflite.Model("./face-net/mobilefacenet.tflite");
console.log("MobileFaceNet TFLite model loaded.");

app.post("/verifyFace", (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: "No image provided" });

    const buffer = Buffer.from(image, "base64");

    // tflite-node expects Uint8Array
    const input = new Uint8Array(buffer);

    const output = model.predict(input);

    res.json({ embedding: Array.from(output) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3291;
app.listen(PORT, () => console.log(`TF Lite server running on port ${PORT}`));
