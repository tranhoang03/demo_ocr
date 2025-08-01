import gradio as gr
import os
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

# Suppress logs
os.environ["GGM_LOG_LEVEL"] = "error"

# Load model & handler (từ Hugging Face Hub)
chat_handler = MoondreamChatHandler.from_pretrained(
    repo_id="tranhoang03/Vintern-fine-tune-ss3",
    filename="*mmproj*.gguf"
)
llm = Llama.from_pretrained(
    repo_id="tranhoang03/Vintern-fine-tune-ss3",
    filename="Vintern-fine-tune-ss3-Q8_0.gguf",
    chat_handler=chat_handler,
    n_ctx=4096,
    log_level="ERROR",
    verbose=False
)

def image_to_data_uri(image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def extract_info(image_path, user_prompt):
    if not image_path:
        return "Vui lòng upload ảnh hóa đơn!"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    uri = image_to_data_uri(image_bytes)
    prompt_text = user_prompt.strip() or "Hãy trích xuất từng thông tin trên hóa đơn này dưới dạng: 'Key': Value"
    messages = [
        {"role": "system", "content": "Bạn là trợ lý đọc và trích xuất thông tin từ hình ảnh."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": uri}}
        ]}
    ]
    try:
        resp = llm.create_chat_completion(messages=messages)
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"

# Custom CSS for sleek UI
custom_css = """
body {
    background: linear-gradient(135deg, #e0f7fa, #80deea);
}
.gradio-container {
    max-width: 800px;
    margin: auto;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    background-color: #ffffff;
}
h1.main-title {
    color: #00695c;
    font-size: 2.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
p.description {
    text-align: center;
    color: #004d40;
    margin-bottom: 1.5rem;
}
.gr-button.primary {
    background-color: #00796b;
    border-color: transparent;
    color: white;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<h1 class='main-title'>📄 Trích xuất thông tin hóa đơn</h1>")
    gr.HTML("<p class='description'>SW</p>")

    with gr.Row():
        with gr.Column(scale=2):
            img_input = gr.Image(
                type="filepath", 
                label="Upload hóa đơn"
            )
            prompt_input = gr.Textbox(
                label="Nhập prompt (tuỳ chọn)", 
                placeholder="Ví dụ: 'Lấy ra số tiền thanh toán và mã hóa đơn'",
                lines=2
            )
            run_btn = gr.Button("🚀 Trích xuất", variant="primary")
        with gr.Column(scale=3):
            output_box = gr.Code(
                label="Kết quả trích xuất", 
                language="json"
            )
    
    run_btn.click(
        fn=extract_info, 
        inputs=[img_input, prompt_input], 
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()
