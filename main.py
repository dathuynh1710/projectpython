import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import show_image_batch_with_predictions

def main():
  # STEP 1: Tạo list ảnh
  IMAGE_FILENAMES = ['pizza.jpg', 'quat.jpg','daibang.jpg','chuoi.jpg','lambo.jpg']
  # STEP 2: Tạo đối tượng ImageClassifier
  base_options = python.BaseOptions(model_asset_path='efficientnet_lite0.tflite')
  options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
  classifier = vision.ImageClassifier.create_from_options(options)

  images = []
  predictions = []
  for image_name in IMAGE_FILENAMES:
      # STEP 3: Load hình ảnh
      image = mp.Image.create_from_file(image_name)

      # STEP 4: Phân loại ảnh
      classification_result = classifier.classify(image)

      # STEP 5: Xử lý kết quả phân loại
      images.append(image)
      top_category = classification_result.classifications[0].categories[0]
      predictions.append(f"{top_category.category_name} ({top_category.score:.2f})")
  show_image_batch_with_predictions(images, predictions)

if __name__ == "__main__":
    main()
