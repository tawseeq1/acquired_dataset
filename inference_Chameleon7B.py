import os
import json
import torch
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# Directories and Paths
MODEL_DIR = "/scratch/22ch10090/models"  # Change this to your model directory
TEST_JSON_PATH = "/scratch/22ch10090/visionlanguage/acquired/acquired/Dataset/test.json"  # Path to test.json
VIDEO_BASE_DIR = "/scratch/22ch10090/visionlanguage/acquired/acquired_dataset"  # Base folder containing video subfolders
OUTPUT_RESULTS = "inference_results.json"  # Output results file

# Load Model and Tokenizer
def load_model(model_dir):
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, device_map="auto")
    
    model.eval()  # Set model to evaluation mode
    # if torch.cuda.is_available():
    #     model.cuda()
    print("Model loaded successfully!")
    return model, processor
# Load Test Data
def load_test_data(json_path):
    print("Loading test data...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    video_paths = [os.path.join(VIDEO_BASE_DIR, item["video_path"]) for item in data]
    return video_paths, data


def infer_videos(model, processor, video_paths, test_data):
    print("Starting inference...")
    results = []
    correct_predictions = 0
    total_predictions = 0

    for video_path, item in tqdm(zip(video_paths, test_data), total=len(video_paths)):
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} does not exist. Skipping...")
            continue
        
        question = item["question"]
        answers = [item["answer1"], item["answer2"]]  # The two possible answers
        
        # Construct the input prompt
        input_prompt = f"{question}, Choose the best option: answer1: {answers[0]} or answer2: {answers[1]}"

        # Prepare input (video + text prompt)
        inputs = processor(videos=video_path, text=input_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_text = processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)[0]
        
        # Determine the predicted label based on output
        predicted_label = 1 if "answer1" in predicted_text else 2
        
        correct_answer_index = 1 if item["correct_answer_key"] == "answer1" else 2
        if predicted_label == correct_answer_index:
            correct_predictions += 1
        
        total_predictions += 1

        result = {
            "video_id": item["video_id"],
            "question": question,
            "predicted_label": predicted_label,
            "correct_answer": correct_answer_index,
        }
        results.append(result)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return results, accuracy

# Save Results to JSON
def save_results(results, output_path):
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully!")

# Main Function
def main():
    model, processor = load_model(MODEL_DIR)
    video_paths, test_data = load_test_data(TEST_JSON_PATH)
    results, accuracy = infer_videos(model, processor, video_paths, test_data)
    save_results(results, OUTPUT_RESULTS)
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()