namespace SampleML
{
    using System;
    using System.IO;
    using Microsoft.ML;
    using Microsoft.ML.Data;

    // Define a data class to hold input features and labels
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        [ColumnName("Label")]
        public string Label;
    }

    // Define a prediction class to hold the predicted label
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }

    public class MLProgram
    {
        static void Main(string[] args)
        {
            // Create a new MLContext
            var context = new MLContext();

            // Load the data from a text file
            var data = context.Data.LoadFromTextFile<IrisData>("iris-data.txt", separatorChar: ',');

            // Split the data into training and testing sets
            var split = context.Data.TrainTestSplit(data);
             
            // Define the data processing pipeline
            var pipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(context.Transforms.NormalizeMinMax("Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("Label"))
                .Append(context.Transforms.Conversion.MapValueToKey("Label"))
                .Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(split.TrainSet);

            // Make predictions on the test data
            var predictions = model.Transform(split.TestSet);

            // Evaluate the model
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");

            // Predict a new sample
            var predictionEngine = context.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);
            var prediction = predictionEngine.Predict(new IrisData
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            });

            Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
        }
    }

}