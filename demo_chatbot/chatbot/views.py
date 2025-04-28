from django.shortcuts import render
from transformers import pipeline

# Khởi tạo pipeline tóm tắt chỉ một lần khi server khởi động
summarizer = pipeline(
    "summarization",
    model="trong269/vit5-vietnamese-text-summarization",
    tokenizer="trong269/vit5-vietnamese-text-summarization",
    framework="pt"  # sử dụng PyTorch
)

def index(request):
    summary = None
    input_text = ""
    if request.method == "POST":
        input_text = request.POST.get("input_text", "").strip()
        if input_text:
            # gọi model tóm tắt
            result = summarizer(
                                    input_text,
                                    max_length=150,
                                    min_length=60,
                                    num_beams=8,
                                    length_penalty=0.8,           # khuyến khích summary ngắn hơn
                                    no_repeat_ngram_size=2,
                                    repetition_penalty=1.5,
                                    early_stopping=True
                                )           # lấy kết quả tóm tắt
            summary = result[0]['summary_text']
    return render(request, "chatbot/index.html", {
        "input_text": input_text,
        "summary": summary
    })
