import natural from "natural";
import fs from "fs";
import express from "express";

const app = express();
const port = 3030;

app.use(express.json());

const classifier = new natural.BayesClassifier();

const trainingData = JSON.parse(fs.readFileSync("training_data.json"));
trainingData.forEach((data) => {
	classifier.addDocument(data.text, data.label);
});

classifier.train();

const testData = JSON.parse(fs.readFileSync("test_data.json"));
let correct = 0;
testData.forEach((data) => {
	const result = classifier.classify(data.text);
	if (result === data.label) {
		correct++;
	}
});

app.post("/email", (req, res) => {
	const { email } = req.body;
	const Accuracy = (correct / testData.length) * 100;
	const classification = classifier.classify(email);
	console.log("Classification:", classification);

	res
		.status(200)
		.send(`Accuracy:${Accuracy}\n Classification:${classification} `);
});

app.listen(port, () => {
	console.log(`Server listening on port ${port}`);
});
