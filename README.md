# Cấu trúc project
- src/cnn: chứa các file liên quan đến CNN, các layer, cấu hình model, training, testing
- src/character_cropper: chứa file pipeline cắt ký tự từ biển số xe thành các ảnh nhỏ hơn
- src/character_classifier: chứa file pipeline nhận dạng ký tự từ các ảnh nhỏ hơn
- src/weight: chứa các file trọng số của các model. Trong đó:
    - character_detector.pt: trọng số của model detector dùng cho cropper
    - character_classifier.v2.pt: trọng số của model classifier dùng cho classifier
- src/tool: chứa các file chạy tool
- src/tool/tools/leo_pipeline.py: chứa file pipelineLeo, là pipeline chính chạy tool, gồm các bước như cắt ảnh ký tự sau đó nhận dạng ký tự.

# Hướng dẫn chạy tool
```bash
python src/tool/app.py
```