# Streaming Data Experiments

This folder contains data and visualizations related to experiments on streaming data for child action recognition using the ST-GCN model. The model is trained using the KS-KSS dataset to classify two action categories: **"Clapping"** and **"Non-Clapping (Other)."** Below is an overview of the files and their respective purposes.

## File Structure

- **Videos (`.avi`)**: Include both skeleton-only visualizations and anonymized grayscale versions (Laplacian-filtered) of the CAR outputs for each hummingbird recording sample.
- **CSV Files (`.csv`)**: Contain results from a sliding window-based classification approach, including softmax probabilities for each class and a detailed breakdown of inference time across different modules in the ST-GCN pipeline.
- **Probability Plots (`<filename>_timestamp.png`)**: Show temporal variation in predicted probabilities for each class throughout the original video.
- **Inference Time Plots (`<filename>_timestamp_inference_time_plot.png`)**: Illustrate how inference time is distributed among different modules of the pipeline for each experiment.

## Example Files

- `buwaneka_Clapping20250418-023836.avi`: Anonymized video of a "Clapping" activity.
- `buwaneka_Clapping20250418-023836_inference_time_plot.png`: Corresponding inference time breakdown for the video above.
- `chenidu_Clapping_GP01_220250418-023836.csv`: Output file containing prediction scores and inference metrics for a "Clapping" activity performed by Chenidu.

## Subfolder: `ruhara`

This subfolder contains a complete set of files for a single video sample, serving as a reference example:

- `ruhara_Clapping_20250418-023836.avi`: Anonymized grayscale video with Laplacian filtering.
- `ruhara_Clapping_20250418-023836.csv`: Sliding window-based prediction outputs and inference time details.
- `ruhara_Clapping_20250418-023836.png`: Probability plot visualizing softmax score variations.
- `ruhara_Clapping_20250418-023836_inference_time_plot.png`: Breakdown of inference times by module.
- `ruhara_Clapping_20250418-025511.avi`: Skeleton-only version of the same recording.
- `ruhara_Clapping_20250418-040024_RGB.mp4`: Partially blurred RGB version of the original input.

> This subfolder is provided for user reference to demonstrate the full pipeline output across multiple visual formats for a single input sample.

## Demo

Click the image below to watch a demo of one of the experiments:

[![Watch the demo anonymized](https://github.com/sankamohotttala/ST_GNN_HAR_DEML/blob/main/streaming_data_experiments/readme_file_related/anon.png)](https://drive.google.com/file/d/12jPlWvXsjNDDQUTewCoEvTwwtdCHCjWo/view)

[![Watch the demo skeleton](https://github.com/sankamohotttala/ST_GNN_HAR_DEML/blob/main/streaming_data_experiments/readme_file_related/skeleton.png)](https://drive.google.com/file/d/1_cEoggHf_zQnJEbngJBPZLHAjcd6v9lqview)

[![Watch the demo RGB](http://github.com/sankamohotttala/ST_GNN_HAR_DEML/blob/main/streaming_data_experiments/readme_file_related/rgb.png)](https://drive.google.com/file/d/1rOPZoFL48aC-fPglSslSFQ1eVSx4ijPG/view)


## How to Use

- **View Activity Videos**: Open the `.avi` or `.mp4` files to observe skeleton-based, grayscale, or RGB representations of the actions.
- **Examine Prediction Outputs**: Use the `.csv` files to review softmax classification results and inference time details.
- **Explore Visualizations**: Analyze `.png` plots to understand model confidence across time and performance breakdown across modules.
- **Assess System Efficiency**: Refer to the `_inference_time_plot.png` files to evaluate which modules are most time-consuming during inference.

---
